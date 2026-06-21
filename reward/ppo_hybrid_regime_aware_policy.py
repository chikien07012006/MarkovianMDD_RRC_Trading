from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .rrc import compute_lambda_rrc


@dataclass(slots=True)
class PPOHybridRegimeAwarePolicyReward:
    lambda_base: float = 0.05
    alpha: float = 0.50
    beta_target: float = 0.02
    beta_turnover: float = 0.0015
    vix_feature_name: str = "vix_zscore_252"
    ret20_feature_name: str = "ret_20d"
    ma_spread_feature_name: str = "ma_spread_5_20"
    clamp_min: float = -3.0
    clamp_max: float = 3.0
    bull_ret20_threshold: float = 0.0
    bull_ma_threshold: float = 0.0
    bear_ret20_threshold: float = 0.0
    bear_ma_threshold: float = 0.0
    stress_threshold: float = 0.80
    crisis_threshold: float = 1.80
    weight_bull: float = 1.0
    weight_bull_stress: float = 0.50
    weight_neutral: float = 0.30
    weight_neutral_negative: float = 0.10
    weight_bear: float = 0.0
    weight_bear_stress: float = -0.10
    weight_bear_crisis: float = -0.80
    return_key: str = "portfolio_log_return"
    drawdown_key: str = "drawdown"
    target_weight_key: str = "target_weight"
    turnover_key: str = "turnover"

    def __post_init__(self) -> None:
        if self.lambda_base < 0:
            raise ValueError("lambda_base must be non-negative.")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if self.beta_target < 0 or self.beta_turnover < 0:
            raise ValueError("Reward coefficients must be non-negative.")
        if self.clamp_min > self.clamp_max:
            raise ValueError("clamp_min must be less than or equal to clamp_max.")
        if self.crisis_threshold < self.stress_threshold:
            raise ValueError("crisis_threshold must be greater than or equal to stress_threshold.")

    def __call__(self, env: Any, transition: dict[str, Any]) -> float:
        del env

        if self.return_key not in transition:
            raise KeyError(f"transition must contain '{self.return_key}'.")
        if self.drawdown_key not in transition:
            raise KeyError(f"transition must contain '{self.drawdown_key}'.")
        if self.target_weight_key not in transition:
            raise KeyError(f"transition must contain '{self.target_weight_key}'.")

        vix_zscore_t = self._resolve_vix_signal(transition)
        lambda_rrc = compute_lambda_rrc(
            vix_zscore_t=vix_zscore_t,
            lambda_base=self.lambda_base,
            alpha=self.alpha,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
        )

        portfolio_log_return = float(transition[self.return_key])
        drawdown_t_plus_1 = float(transition[self.drawdown_key])
        target_weight = float(transition[self.target_weight_key])
        turnover = float(transition.get(self.turnover_key, 0.0))

        previous_nav = float(transition.get("previous_nav", transition.get("nav", 1.0)))
        previous_running_peak = float(
            transition.get("previous_running_peak", transition.get("running_peak", previous_nav))
        )
        previous_drawdown = 0.0
        if previous_running_peak > 0:
            previous_drawdown = max(0.0, previous_running_peak - previous_nav) / previous_running_peak
        drawdown_increase = max(drawdown_t_plus_1 - previous_drawdown, 0.0)
        drawdown_penalty = lambda_rrc * drawdown_increase

        desired_weight = self._compute_desired_weight(transition, vix_zscore_t)
        target_penalty = self.beta_target * ((target_weight - desired_weight) ** 2)
        turnover_penalty = self.beta_turnover * turnover

        reward = portfolio_log_return - drawdown_penalty - target_penalty - turnover_penalty

        transition["vix_zscore_t"] = float(vix_zscore_t)
        transition["lambda_rrc"] = lambda_rrc
        transition["previous_drawdown"] = previous_drawdown
        transition["drawdown_increase"] = drawdown_increase
        transition["drawdown_penalty"] = drawdown_penalty
        transition["desired_weight"] = desired_weight
        transition["target_penalty"] = target_penalty
        transition["turnover_penalty"] = turnover_penalty
        transition["reward"] = reward

        return float(reward)

    def _resolve_vix_signal(self, transition: dict[str, Any]) -> float:
        if "vix_zscore_t" in transition:
            return float(transition["vix_zscore_t"])

        if "market_features_t" in transition and self.vix_feature_name in transition["market_features_t"]:
            return float(transition["market_features_t"][self.vix_feature_name])

        if (
            "previous_observation_dict" in transition
            and self.vix_feature_name in transition["previous_observation_dict"]
        ):
            return float(transition["previous_observation_dict"][self.vix_feature_name])

        raise KeyError(
            f"transition must contain '{self.vix_feature_name}' information for Risk-Regime Conditioning."
        )

    def _compute_desired_weight(self, transition: dict[str, Any], vix_zscore_t: float) -> float:
        market_features = transition.get("market_features_t", {})
        previous_observation_dict = transition.get("previous_observation_dict", {})

        ret_20d = float(
            market_features.get(
                self.ret20_feature_name,
                previous_observation_dict.get(self.ret20_feature_name, 0.0),
            )
        )
        ma_spread = float(
            market_features.get(
                self.ma_spread_feature_name,
                previous_observation_dict.get(self.ma_spread_feature_name, 0.0),
            )
        )
        bullish = ret_20d > self.bull_ret20_threshold and ma_spread > self.bull_ma_threshold
        bearish = ret_20d < self.bear_ret20_threshold and ma_spread < self.bear_ma_threshold
        stress = float(vix_zscore_t) > self.stress_threshold
        crisis = float(vix_zscore_t) > self.crisis_threshold

        if bullish and not stress:
            return float(self.weight_bull)
        if bullish and stress:
            return float(self.weight_bull_stress)
        if bearish and crisis:
            return float(self.weight_bear_crisis)
        if bearish and stress:
            return float(self.weight_bear_stress)
        if bearish:
            return float(self.weight_bear)
        if ret_20d < 0 or ma_spread < 0:
            return float(self.weight_neutral_negative)
        return float(self.weight_neutral)


def build_ppo_hybrid_regime_aware_policy_reward(
    lambda_base: float = 0.05,
    alpha: float = 0.50,
    beta_target: float = 0.02,
    beta_turnover: float = 0.0015,
    vix_feature_name: str = "vix_zscore_252",
    ret20_feature_name: str = "ret_20d",
    ma_spread_feature_name: str = "ma_spread_5_20",
    clamp_min: float = -3.0,
    clamp_max: float = 3.0,
    bull_ret20_threshold: float = 0.0,
    bull_ma_threshold: float = 0.0,
    bear_ret20_threshold: float = 0.0,
    bear_ma_threshold: float = 0.0,
    stress_threshold: float = 0.80,
    crisis_threshold: float = 1.80,
    weight_bull: float = 1.0,
    weight_bull_stress: float = 0.50,
    weight_neutral: float = 0.30,
    weight_neutral_negative: float = 0.10,
    weight_bear: float = 0.0,
    weight_bear_stress: float = -0.10,
    weight_bear_crisis: float = -0.80,
    return_key: str = "portfolio_log_return",
    drawdown_key: str = "drawdown",
    target_weight_key: str = "target_weight",
    turnover_key: str = "turnover",
) -> PPOHybridRegimeAwarePolicyReward:
    return PPOHybridRegimeAwarePolicyReward(
        lambda_base=lambda_base,
        alpha=alpha,
        beta_target=beta_target,
        beta_turnover=beta_turnover,
        vix_feature_name=vix_feature_name,
        ret20_feature_name=ret20_feature_name,
        ma_spread_feature_name=ma_spread_feature_name,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        bull_ret20_threshold=bull_ret20_threshold,
        bull_ma_threshold=bull_ma_threshold,
        bear_ret20_threshold=bear_ret20_threshold,
        bear_ma_threshold=bear_ma_threshold,
        stress_threshold=stress_threshold,
        crisis_threshold=crisis_threshold,
        weight_bull=weight_bull,
        weight_bull_stress=weight_bull_stress,
        weight_neutral=weight_neutral,
        weight_neutral_negative=weight_neutral_negative,
        weight_bear=weight_bear,
        weight_bear_stress=weight_bear_stress,
        weight_bear_crisis=weight_bear_crisis,
        return_key=return_key,
        drawdown_key=drawdown_key,
        target_weight_key=target_weight_key,
        turnover_key=turnover_key,
    )


__all__ = [
    "PPOHybridRegimeAwarePolicyReward",
    "build_ppo_hybrid_regime_aware_policy_reward",
]
