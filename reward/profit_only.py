from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ProfitOnlyReward:
    return_key: str = "portfolio_log_return"

    def __call__(self, env: Any, transition: dict[str, Any]) -> float:
        del env

        if self.return_key not in transition:
            raise KeyError(f"transition must contain '{self.return_key}'.")

        reward = float(transition[self.return_key])
        transition["reward"] = reward
        return reward


def build_profit_only_reward(
    return_key: str = "portfolio_log_return",
) -> ProfitOnlyReward:
    return ProfitOnlyReward(return_key=return_key)
