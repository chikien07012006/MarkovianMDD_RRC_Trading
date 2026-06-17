from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VariancePenalizedReward:
    lambda_penalty: float = 1.0
    return_key: str = "portfolio_log_return"
    count: int = field(init=False, default=0, repr=False)
    mean_return: float = field(init=False, default=0.0, repr=False)
    m2_return: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.lambda_penalty < 0:
            raise ValueError("lambda_penalty must be non-negative.")

    def reset(self, env: Any | None = None) -> None:
        del env
        self.count = 0
        self.mean_return = 0.0
        self.m2_return = 0.0

    def __call__(self, env: Any, transition: dict[str, Any]) -> float:
        del env

        if self.return_key not in transition:
            raise KeyError(f"transition must contain '{self.return_key}'.")

        portfolio_log_return = float(transition[self.return_key])

        self.count += 1
        delta = portfolio_log_return - self.mean_return
        self.mean_return += delta / self.count
        delta_after_update = portfolio_log_return - self.mean_return
        self.m2_return += delta * delta_after_update

        portfolio_variance = self.m2_return / self.count
        variance_penalty = self.lambda_penalty * portfolio_variance
        reward = portfolio_log_return - variance_penalty

        transition["portfolio_variance"] = portfolio_variance
        transition["variance_penalty_lambda"] = self.lambda_penalty
        transition["variance_penalty"] = variance_penalty
        transition["variance_penalty_count"] = self.count
        transition["variance_penalty_mean_return"] = self.mean_return
        transition["reward"] = reward

        return float(reward)


def build_variance_penalized_reward(
    lambda_penalty: float = 1.0,
    return_key: str = "portfolio_log_return",
) -> VariancePenalizedReward:
    return VariancePenalizedReward(
        lambda_penalty=lambda_penalty,
        return_key=return_key,
    )
