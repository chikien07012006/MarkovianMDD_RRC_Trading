from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from reward import (
    build_differential_sharpe_reward,
    build_markovian_mdd_rrc_reward,
    build_markovian_mdd_static_reward,
    build_variance_penalized_reward,
)


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
        reward_mode: str = "default",
        lambda_base: float = 1.0,
        alpha: float = 0.0,
        reward_kwargs: dict[str, Any] | None = None,
        vix_feature_name: str = "vix_zscore_252",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.initial_nav = float(initial_nav)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.min_nav = 1e-12
        self.vix_feature_name = vix_feature_name

        self.feature_columns = feature_columns or self.DEFAULT_FEATURE_COLUMNS
        self.data = self._load_data(data)
        self._validate_data()
        self.reward_fn = self._resolve_reward_fn(
            reward_fn=reward_fn,
            reward_mode=reward_mode,
            lambda_base=lambda_base,
            alpha=alpha,
            reward_kwargs=reward_kwargs,
            vix_feature_name=vix_feature_name,
        )

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

        if self.vix_feature_name not in self.data.columns:
            raise ValueError(f"Dataset is missing required RRC feature column: {self.vix_feature_name}")

    def _resolve_reward_fn(
        self,
        reward_fn: RewardFn | None,
        reward_mode: str,
        lambda_base: float,
        alpha: float,
        reward_kwargs: dict[str, Any] | None,
        vix_feature_name: str,
    ) -> RewardFn | None:
        if reward_fn is not None:
            return reward_fn

        normalized_reward_mode = reward_mode.strip().lower()
        if normalized_reward_mode in {"default", "log_return", "r0"}:
            return None

        resolved_reward_kwargs = dict(reward_kwargs or {})
        resolved_reward_kwargs.setdefault("lambda_penalty", lambda_base)
        resolved_reward_kwargs.setdefault("lambda_base", lambda_base)
        resolved_reward_kwargs.setdefault("alpha", alpha)
        resolved_reward_kwargs.setdefault("vix_feature_name", vix_feature_name)

        if normalized_reward_mode in {"variance_penalized", "variance_penalized_return", "r1"}:
            return build_variance_penalized_reward(
                lambda_penalty=float(resolved_reward_kwargs["lambda_penalty"]),
                return_key=str(resolved_reward_kwargs.get("return_key", "portfolio_log_return")),
            )

        if normalized_reward_mode in {"differential_sharpe", "dsr", "r2"}:
            return build_differential_sharpe_reward(
                eta=float(resolved_reward_kwargs.get("eta", 0.01)),
                return_key=str(resolved_reward_kwargs.get("return_key", "portfolio_log_return")),
                epsilon=float(resolved_reward_kwargs.get("epsilon", 1e-12)),
            )

        if normalized_reward_mode in {"markovian_mdd", "markovian_mdd_static", "r3"}:
            return build_markovian_mdd_static_reward(
                lambda_penalty=float(resolved_reward_kwargs["lambda_penalty"]),
                return_key=str(resolved_reward_kwargs.get("return_key", "portfolio_log_return")),
                drawdown_key=str(resolved_reward_kwargs.get("drawdown_key", "drawdown")),
            )

        if normalized_reward_mode in {"markovian_mdd_rrc", "r4"}:
            return build_markovian_mdd_rrc_reward(
                lambda_base=float(resolved_reward_kwargs["lambda_base"]),
                alpha=float(resolved_reward_kwargs["alpha"]),
                vix_feature_name=str(resolved_reward_kwargs["vix_feature_name"]),
                clamp_min=float(resolved_reward_kwargs.get("clamp_min", -3.0)),
                clamp_max=float(resolved_reward_kwargs.get("clamp_max", 3.0)),
                return_key=str(resolved_reward_kwargs.get("return_key", "portfolio_log_return")),
                drawdown_key=str(resolved_reward_kwargs.get("drawdown_key", "drawdown")),
            )

        raise ValueError(
            "reward_mode must be one of "
            "{'default', 'log_return', 'variance_penalized', 'differential_sharpe', "
            "'markovian_mdd', 'markovian_mdd_rrc', 'r0', 'r1', 'r2', 'r3', 'r4'} "
            "when reward_fn is not provided."
        )

    def _reset_reward_fn_state(self) -> None:
        if self.reward_fn is None:
            return

        reset_method = getattr(self.reward_fn, "reset", None)
        if callable(reset_method):
            reset_method(self)

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

    def _get_market_feature_dict(self, step: int) -> dict[str, float]:
        return {
            feature_name: float(self.data.iloc[step][feature_name]) for feature_name in self.feature_columns
        }

    def _get_observation_dict(self, step: int) -> dict[str, float]:
        observation_dict = self._get_market_feature_dict(step)
        observation_dict.update(
            {
                "current_cash_ratio": float(self.current_cash_ratio),
                "current_weight": float(self.current_weight),
                "unrealized_pnl": float(self._compute_unrealized_pnl()),
                "running_peak": float(self.running_peak),
            }
        )
        return observation_dict

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
        self._reset_reward_fn_state()

        observation = self._get_observation()
        info = self._build_info()
        return observation, info

    def step(self, action: np.ndarray | float | list[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.terminated:
            raise RuntimeError("Episode is done. Call reset() before calling step() again.")

        target_weight = self._parse_action(action)
        previous_observation = self._get_observation().copy()
        previous_observation_dict = self._get_observation_dict(self.current_step)
        market_features_t = self._get_market_feature_dict(self.current_step)
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
            "previous_observation_dict": previous_observation_dict,
            "market_features_t": market_features_t,
            "vix_zscore_t": float(market_features_t[self.vix_feature_name]),
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
