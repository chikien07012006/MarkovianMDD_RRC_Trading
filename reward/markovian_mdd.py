from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .rrc import compute_lambda_rrc


@dataclass(slots=True)
class MarkovianMDDRRCReward:
    lambda_base: float = 1.0
    alpha: float = 0.0
    vix_feature_name: str = "vix_zscore_252"
    clamp_min: float = -3.0
    clamp_max: float = 3.0
    return_key: str = "portfolio_log_return"
    drawdown_key: str = "drawdown"

    def __post_init__(self) -> None:
        if self.lambda_base < 0:
            raise ValueError("lambda_base must be non-negative.")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if self.clamp_min > self.clamp_max:
            raise ValueError("clamp_min must be less than or equal to clamp_max.")

    def __call__(self, env: Any, transition: dict[str, Any]) -> float:
        del env

        if self.return_key not in transition:
            raise KeyError(f"transition must contain '{self.return_key}'.")
        if self.drawdown_key not in transition:
            raise KeyError(f"transition must contain '{self.drawdown_key}'.")

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
        drawdown_penalty = lambda_rrc * drawdown_t_plus_1
        reward = portfolio_log_return - drawdown_penalty

        transition["vix_zscore_t"] = float(vix_zscore_t)
        transition["lambda_rrc"] = lambda_rrc
        transition["drawdown_t_plus_1"] = drawdown_t_plus_1
        transition["drawdown_penalty"] = drawdown_penalty
        transition["reward"] = reward

        return float(reward)

    def _resolve_vix_signal(self, transition: dict[str, Any]) -> float:
        if "vix_zscore_t" in transition:
            return float(transition["vix_zscore_t"])

        if "market_features_t" in transition and self.vix_feature_name in transition["market_features_t"]:
            return float(transition["market_features_t"][self.vix_feature_name])

        if "previous_observation_dict" in transition and self.vix_feature_name in transition["previous_observation_dict"]:
            return float(transition["previous_observation_dict"][self.vix_feature_name])

        raise KeyError(
            f"transition must contain '{self.vix_feature_name}' information for Risk-Regime Conditioning."
        )


def build_markovian_mdd_rrc_reward(
    lambda_base: float = 1.0,
    alpha: float = 0.0,
    vix_feature_name: str = "vix_zscore_252",
    clamp_min: float = -3.0,
    clamp_max: float = 3.0,
    return_key: str = "portfolio_log_return",
    drawdown_key: str = "drawdown",
) -> MarkovianMDDRRCReward:
    return MarkovianMDDRRCReward(
        lambda_base=lambda_base,
        alpha=alpha,
        vix_feature_name=vix_feature_name,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        return_key=return_key,
        drawdown_key=drawdown_key,
    )


MarkovianMDDReward = MarkovianMDDRRCReward
build_markovian_mdd_reward = build_markovian_mdd_rrc_reward
