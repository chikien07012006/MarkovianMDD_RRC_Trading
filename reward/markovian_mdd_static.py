from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class StaticMarkovianMDDReward:
    lambda_penalty: float = 1.0
    return_key: str = "portfolio_log_return"
    drawdown_key: str = "drawdown"

    def __post_init__(self) -> None:
        if self.lambda_penalty < 0:
            raise ValueError("lambda_penalty must be non-negative.")

    def __call__(self, env: Any, transition: dict[str, Any]) -> float:
        del env

        if self.return_key not in transition:
            raise KeyError(f"transition must contain '{self.return_key}'.")
        if self.drawdown_key not in transition:
            raise KeyError(f"transition must contain '{self.drawdown_key}'.")

        portfolio_log_return = float(transition[self.return_key])
        drawdown_t = float(transition[self.drawdown_key])
        drawdown_penalty = self.lambda_penalty * drawdown_t
        reward = portfolio_log_return - drawdown_penalty

        transition["drawdown_t_plus_1"] = drawdown_t
        transition["mdd_penalty_lambda"] = self.lambda_penalty
        transition["drawdown_penalty"] = drawdown_penalty
        transition["reward"] = reward

        return float(reward)


def build_markovian_mdd_static_reward(
    lambda_penalty: float = 1.0,
    return_key: str = "portfolio_log_return",
    drawdown_key: str = "drawdown",
) -> StaticMarkovianMDDReward:
    return StaticMarkovianMDDReward(
        lambda_penalty=lambda_penalty,
        return_key=return_key,
        drawdown_key=drawdown_key,
    )
