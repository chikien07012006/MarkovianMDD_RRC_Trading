from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DifferentialSharpeReward:
    eta: float = 0.01
    return_key: str = "portfolio_log_return"
    epsilon: float = 1e-12
    mean_ema: float = field(init=False, default=0.0, repr=False)
    second_moment_ema: float = field(init=False, default=0.0, repr=False)
    initialized: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.eta <= 1:
            raise ValueError("eta must be in the interval (0, 1].")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")

    def reset(self, env: Any | None = None) -> None:
        del env
        self.mean_ema = 0.0
        self.second_moment_ema = 0.0
        self.initialized = False

    def __call__(self, env: Any, transition: dict[str, Any]) -> float:
        del env

        if self.return_key not in transition:
            raise KeyError(f"transition must contain '{self.return_key}'.")

        portfolio_log_return = float(transition[self.return_key])
        previous_mean = self.mean_ema
        previous_second_moment = self.second_moment_ema

        delta_mean = portfolio_log_return - previous_mean
        delta_second_moment = (portfolio_log_return**2) - previous_second_moment
        previous_variance = max(
            previous_second_moment - (previous_mean**2),
            0.0,
        )

        if not self.initialized or previous_variance <= self.epsilon:
            differential_sharpe = 0.0
        else:
            differential_sharpe = (
                (previous_second_moment * delta_mean)
                - (0.5 * previous_mean * delta_second_moment)
            ) / ((previous_variance + self.epsilon) ** 1.5)

        self.mean_ema = previous_mean + (self.eta * delta_mean)
        self.second_moment_ema = previous_second_moment + (self.eta * delta_second_moment)
        self.initialized = True

        ema_variance = max(self.second_moment_ema - (self.mean_ema**2), 0.0)
        ema_std = math.sqrt(ema_variance)

        transition["differential_sharpe"] = differential_sharpe
        transition["dsr_eta"] = self.eta
        transition["return_ema_mean"] = self.mean_ema
        transition["return_ema_variance"] = ema_variance
        transition["return_ema_std"] = ema_std
        transition["reward"] = differential_sharpe

        return float(differential_sharpe)


def build_differential_sharpe_reward(
    eta: float = 0.01,
    return_key: str = "portfolio_log_return",
    epsilon: float = 1e-12,
) -> DifferentialSharpeReward:
    return DifferentialSharpeReward(
        eta=eta,
        return_key=return_key,
        epsilon=epsilon,
    )
