from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


RewardFn = Callable[["TradingEnv", dict[str, Any]], float]


class TradingEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    DEFAULT_FEATURE_COLUMNS = [
        "log_return",
        "sma_ratio",
        "rsi_14",
        "bollinger_band_width",
        "vix_zscore_252",
    ]
    REQUIRED_PRICE_COLUMNS = ["date", "close"]

    def __init__(
        self,
        data: pd.DataFrame | str | Path,
        feature_columns: list[str] | None = None,
        initial_nav: float = 1.0,
        transaction_cost_rate: float = 0.001,
        reward_fn: RewardFn | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.initial_nav = float(initial_nav)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.reward_fn = reward_fn
        self.min_nav = 1e-12

        self.feature_columns = feature_columns or self.DEFAULT_FEATURE_COLUMNS
        self.data = self._load_data(data)
        self._validate_data()

        self.close_prices = self.data["close"].to_numpy(dtype=np.float64)
        self.dates = self.data["date"].dt.strftime("%Y-%m-%d").to_numpy()
        self.market_features = self.data[self.feature_columns].to_numpy(dtype=np.float32)

        observation_dim = len(self.feature_columns) + 4
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )

        self.last_index = len(self.data) - 1

        self.current_step = 0
        self.nav = self.initial_nav
        self.running_peak = self.initial_nav
        self.current_weight = 0.0
        self.current_cash_ratio = 1.0
        self.terminated = False

    def _load_data(self, data: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            frame = pd.read_csv(data, parse_dates=["date"])

        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date").reset_index(drop=True)
        return frame

    def _validate_data(self) -> None:
        required_columns = set(self.REQUIRED_PRICE_COLUMNS + self.feature_columns)
        missing_columns = sorted(required_columns.difference(self.data.columns))
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

        if len(self.data) < 2:
            raise ValueError("TradingEnv requires at least two rows of market data.")

        if self.initial_nav <= 0:
            raise ValueError("initial_nav must be positive.")

        if self.transaction_cost_rate < 0:
            raise ValueError("transaction_cost_rate must be non-negative.")

    def _parse_action(self, action: np.ndarray | float | list[float]) -> float:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != 1:
            raise ValueError("Action must be a scalar or shape-(1,) array.")
        return float(np.clip(action_array[0], -1.0, 1.0))

    def _compute_unrealized_pnl(self) -> float:
        return (self.nav - self.initial_nav) / self.initial_nav

    def _get_observation(self) -> np.ndarray:
        portfolio_state = np.array(
            [
                self.current_cash_ratio,
                self.current_weight,
                self._compute_unrealized_pnl(),
                self.running_peak,
            ],
            dtype=np.float32,
        )
        return np.concatenate([self.market_features[self.current_step], portfolio_state]).astype(
            np.float32
        )

    def _build_info(self) -> dict[str, Any]:
        return {
            "date": self.dates[self.current_step],
            "step": self.current_step,
            "nav": self.nav,
            "running_peak": self.running_peak,
            "drawdown": (self.running_peak - self.nav) / self.running_peak,
            "current_weight": self.current_weight,
            "cash_ratio": self.current_cash_ratio,
            "unrealized_pnl": self._compute_unrealized_pnl(),
        }

    def _default_reward(self, transition: dict[str, Any]) -> float:
        return float(transition["portfolio_log_return"])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self.current_step = 0
        self.nav = self.initial_nav
        self.running_peak = self.initial_nav
        self.current_weight = 0.0
        self.current_cash_ratio = 1.0
        self.terminated = False

        observation = self._get_observation()
        info = self._build_info()
        return observation, info

    def step(self, action: np.ndarray | float | list[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.terminated:
            raise RuntimeError("Episode is done. Call reset() before calling step() again.")

        target_weight = self._parse_action(action)
        previous_observation = self._get_observation().copy()
        previous_running_peak = self.running_peak

        previous_step = self.current_step
        next_step = previous_step + 1

        current_close = self.close_prices[previous_step]
        next_close = self.close_prices[next_step]
        previous_nav = self.nav
        previous_weight = self.current_weight

        turnover = abs(target_weight - previous_weight)
        transaction_cost = previous_nav * self.transaction_cost_rate * turnover
        nav_after_cost = max(previous_nav - transaction_cost, self.min_nav)

        asset_return = (next_close / current_close) - 1.0
        portfolio_simple_return = target_weight * asset_return
        gross_next_nav = nav_after_cost * (1.0 + portfolio_simple_return)
        next_nav = max(gross_next_nav, self.min_nav)

        if gross_next_nav <= self.min_nav:
            drifted_weight = 0.0
        else:
            risky_asset_value = nav_after_cost * target_weight * (1.0 + asset_return)
            drifted_weight = risky_asset_value / gross_next_nav

        self.current_step = next_step
        self.nav = next_nav
        self.running_peak = max(self.running_peak, self.nav)
        self.current_weight = float(drifted_weight)
        self.current_cash_ratio = 1.0 - self.current_weight
        self.terminated = self.current_step >= self.last_index

        drawdown = (self.running_peak - self.nav) / self.running_peak
        portfolio_log_return = math.log(self.nav / previous_nav)

        observation = self._get_observation()
        transition = {
            "previous_step": previous_step,
            "current_step": self.current_step,
            "previous_date": self.dates[previous_step],
            "current_date": self.dates[self.current_step],
            "previous_observation": previous_observation,
            "previous_running_peak": previous_running_peak,
            "previous_nav": previous_nav,
            "nav_after_cost": nav_after_cost,
            "nav": self.nav,
            "running_peak": self.running_peak,
            "drawdown": drawdown,
            "transaction_cost": transaction_cost,
            "transaction_cost_rate": self.transaction_cost_rate,
            "turnover": turnover,
            "current_close": current_close,
            "next_close": next_close,
            "asset_return": asset_return,
            "portfolio_simple_return": portfolio_simple_return,
            "portfolio_log_return": portfolio_log_return,
            "previous_weight": previous_weight,
            "target_weight": target_weight,
            "current_weight": self.current_weight,
            "cash_ratio": self.current_cash_ratio,
            "unrealized_pnl": self._compute_unrealized_pnl(),
            "observation": observation.copy(),
            "next_observation": observation.copy(),
            "terminated": self.terminated,
        }

        reward = (
            self._default_reward(transition)
            if self.reward_fn is None
            else float(self.reward_fn(self, transition))
        )

        info = self._build_info()
        info.update(transition)

        return observation, reward, self.terminated, False, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        info = self._build_info()
        print(
            f"date={info['date']} step={info['step']} nav={info['nav']:.6f} "
            f"weight={info['current_weight']:.4f} cash_ratio={info['cash_ratio']:.4f} "
            f"drawdown={info['drawdown']:.4f}"
        )

    def close(self) -> None:
        return None
