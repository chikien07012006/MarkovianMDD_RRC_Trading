from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "markovian_mdd_rrc_trading_mpl"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.metrics import compute_performance_metrics
from env.trading_env import TradingEnv


TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spy_vix_indicators_train.csv"
VALIDATION_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spy_vix_indicators_validation.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spy_vix_indicators_test.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models" / "rl_baselines"
RL_TABLES_DIR = TABLES_DIR / "rl_baselines"
TRANSACTION_COST_RATE = 0.001


@dataclass(slots=True)
class RLBaselineConfig:
    name: str
    algorithm: str
    reward_mode: str
    total_timesteps: int = 100_000
    seed: int = 42
    device: str = "cpu"
    initial_capital: float = 10_000.0
    transaction_cost_rate: float = TRANSACTION_COST_RATE
    lambda_base: float = 1.0
    alpha: float = 0.0
    reward_kwargs: dict[str, Any] = field(default_factory=dict)
    train_data_path: Path = TRAIN_DATA_PATH
    validation_data_path: Path = VALIDATION_DATA_PATH
    test_data_path: Path = TEST_DATA_PATH
    model_dir: Path = MODELS_DIR
    figures_dir: Path = FIGURES_DIR
    tables_dir: Path = RL_TABLES_DIR
    algo_kwargs: dict[str, Any] = field(default_factory=dict)
    deterministic_eval: bool = True
    retrain_if_exists: bool = False


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def load_market_data(data_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path, parse_dates=["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    required_columns = set(["date", "close"] + TradingEnv.DEFAULT_FEATURE_COLUMNS)
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    cleaned = frame.dropna(
        subset=["close"] + TradingEnv.DEFAULT_FEATURE_COLUMNS
    ).reset_index(drop=True)
    if len(cleaned) < 2:
        raise ValueError(f"Dataset '{data_path}' does not contain enough valid rows after dropping NaNs.")
    return cleaned


def make_trading_env(
    data: pd.DataFrame,
    reward_mode: str,
    *,
    reward_kwargs: dict[str, Any] | None = None,
    lambda_base: float = 1.0,
    alpha: float = 0.0,
    initial_capital: float = 10_000.0,
    transaction_cost_rate: float = TRANSACTION_COST_RATE,
) -> TradingEnv:
    return TradingEnv(
        data=data,
        initial_nav=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
        reward_mode=reward_mode,
        lambda_base=lambda_base,
        alpha=alpha,
        reward_kwargs=reward_kwargs,
    )


def build_algorithm(config: RLBaselineConfig, env: TradingEnv) -> BaseAlgorithm:
    algorithm_name = config.algorithm.strip().lower()
    shared_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "seed": config.seed,
        "device": config.device,
        "verbose": 0,
        "policy_kwargs": dict(net_arch=[64, 64]),
    }

    if algorithm_name == "ppo":
        default_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
        }
        default_kwargs.update(config.algo_kwargs)
        return PPO(**shared_kwargs, **default_kwargs)

    if algorithm_name == "sac":
        default_kwargs = {
            "learning_rate": 3e-4,
            "buffer_size": 50_000,
            "learning_starts": 1_000,
            "batch_size": 64,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": 0.01,
        }
        default_kwargs.update(config.algo_kwargs)
        return SAC(**shared_kwargs, **default_kwargs)

    raise ValueError(f"Unsupported algorithm '{config.algorithm}'.")


def train_model(config: RLBaselineConfig) -> Path:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_dir / config.name
    model_zip_path = model_path.with_suffix(".zip")
    if model_zip_path.exists() and not config.retrain_if_exists:
        return model_zip_path

    train_data = load_market_data(config.train_data_path)
    train_env = make_trading_env(
        data=train_data,
        reward_mode=config.reward_mode,
        reward_kwargs=config.reward_kwargs,
        lambda_base=config.lambda_base,
        alpha=config.alpha,
        initial_capital=config.initial_capital,
        transaction_cost_rate=config.transaction_cost_rate,
    )

    model = build_algorithm(config, train_env)
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)

    model.save(str(model_path))
    train_env.close()

    metadata = _to_jsonable(asdict(config))
    metadata_path = config.tables_dir / f"{config.name}_config.json"
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return model_zip_path


def load_trained_model(config: RLBaselineConfig, model_path: str | Path) -> BaseAlgorithm:
    algorithm_name = config.algorithm.strip().lower()
    if algorithm_name == "ppo":
        return PPO.load(str(model_path), device=config.device)
    if algorithm_name == "sac":
        return SAC.load(str(model_path), device=config.device)
    raise ValueError(f"Unsupported algorithm '{config.algorithm}'.")


def evaluate_saved_model(
    config: RLBaselineConfig,
    model_path: str | Path,
) -> dict[str, pd.Series | pd.DataFrame | dict[str, float] | str]:
    test_data = load_market_data(config.test_data_path)
    test_env = make_trading_env(
        data=test_data,
        reward_mode=config.reward_mode,
        reward_kwargs=config.reward_kwargs,
        lambda_base=config.lambda_base,
        alpha=config.alpha,
        initial_capital=config.initial_capital,
        transaction_cost_rate=config.transaction_cost_rate,
    )
    model = load_trained_model(config, model_path)

    observation, info = test_env.reset(seed=config.seed)
    dates = [pd.Timestamp(info["date"])]
    portfolio_values = [float(info["nav"])]
    daily_returns = [0.0]
    turnover = [0.0]
    weights = [float(info["current_weight"])]
    cash_ratios = [float(info["cash_ratio"])]
    rewards = [0.0]

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=config.deterministic_eval)
        observation, reward, terminated, truncated, info = test_env.step(action)

        dates.append(pd.Timestamp(info["date"]))
        portfolio_values.append(float(info["nav"]))
        daily_returns.append(float(info["nav"] / info["previous_nav"] - 1.0))
        turnover.append(float(info["turnover"]))
        weights.append(float(info["current_weight"]))
        cash_ratios.append(float(info["cash_ratio"]))
        rewards.append(float(reward))

    index = pd.DatetimeIndex(dates, name="date")
    portfolio_value_series = pd.Series(portfolio_values, index=index, name="portfolio_value")
    daily_returns_series = pd.Series(daily_returns, index=index, name="daily_returns")
    turnover_series = pd.Series(turnover, index=index, name="turnover")

    diagnostics = pd.DataFrame(
        {
            "portfolio_value": portfolio_value_series,
            "daily_returns": daily_returns_series,
            "turnover": turnover_series,
            "weight_spy": pd.Series(weights, index=index),
            "weight_cash": pd.Series(cash_ratios, index=index),
            "reward": pd.Series(rewards, index=index),
        }
    )

    metrics = compute_performance_metrics(
        portfolio_value=portfolio_value_series,
        daily_returns=daily_returns_series,
        turnover=turnover_series,
    )
    test_env.close()

    return {
        "baseline": config.name,
        "portfolio_value": portfolio_value_series,
        "daily_returns": daily_returns_series,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "model_path": str(model_path),
    }


def save_single_result(
    config: RLBaselineConfig,
    result: dict[str, pd.Series | pd.DataFrame | dict[str, float] | str],
) -> tuple[Path, Path]:
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = result["diagnostics"].copy()
    diagnostics.insert(0, "baseline", config.name)
    diagnostics.insert(1, "date", diagnostics.index)

    portfolios_path = config.tables_dir / f"{config.name}_portfolios.csv"
    metrics_path = config.tables_dir / f"{config.name}_metrics.json"

    diagnostics.reset_index(drop=True).to_csv(portfolios_path, index=False)

    payload = {
        "baseline": config.name,
        "algorithm": config.algorithm,
        "reward_mode": config.reward_mode,
        "model_path": result["model_path"],
        "metrics": {metric_name: float(metric_value) for metric_name, metric_value in result["metrics"].items()},
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return portfolios_path, metrics_path


def plot_rl_equity_curves(
    baseline_results: dict[str, dict[str, pd.Series | pd.DataFrame | dict[str, float] | str]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    label_map = {
        "ppo_profit_only": "PPO + Profit Only",
        "sac_profit_only": "SAC + Profit Only",
        "ppo_variance_penalized": "PPO + Variance Penalized",
        "ppo_markovian_mdd_static": "PPO + Markovian MDD (Static)",
    }
    color_map = {
        "ppo_profit_only": "#1f77b4",
        "sac_profit_only": "#ff7f0e",
        "ppo_variance_penalized": "#2ca02c",
        "ppo_markovian_mdd_static": "#d62728",
    }

    for baseline_name, result in baseline_results.items():
        series = result["portfolio_value"]
        ax.plot(
            series.index,
            series.values,
            linewidth=2.0,
            color=color_map.get(baseline_name),
            label=label_map.get(baseline_name, baseline_name),
        )

    ax.set_title("RL Baselines Equity Curve Comparison", fontsize=14, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_aggregate_results(
    baseline_results: dict[str, dict[str, pd.Series | pd.DataFrame | dict[str, float] | str]],
    tables_dir: str | Path = RL_TABLES_DIR,
) -> tuple[Path, Path, Path]:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    diagnostic_frames: list[pd.DataFrame] = []

    for baseline_name, result in baseline_results.items():
        row = {"baseline": baseline_name}
        row.update(result["metrics"])
        metric_rows.append(row)

        diagnostics = result["diagnostics"].copy()
        diagnostics.insert(0, "baseline", baseline_name)
        diagnostics.insert(1, "date", diagnostics.index)
        diagnostic_frames.append(diagnostics.reset_index(drop=True))

    metrics_csv_path = tables_dir / "rl_baselines_metrics.csv"
    metrics_json_path = tables_dir / "rl_baselines_metrics.json"
    portfolios_csv_path = tables_dir / "rl_baselines_portfolios.csv"

    pd.DataFrame(metric_rows).to_csv(metrics_csv_path, index=False)
    pd.concat(diagnostic_frames, ignore_index=True).to_csv(portfolios_csv_path, index=False)

    json_payload = {
        baseline_name: {
            "metrics": {metric_name: float(metric_value) for metric_name, metric_value in result["metrics"].items()},
            "model_path": result["model_path"],
        }
        for baseline_name, result in baseline_results.items()
    }
    metrics_json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    return metrics_csv_path, metrics_json_path, portfolios_csv_path


def run_single_rl_baseline(
    config: RLBaselineConfig,
) -> dict[str, pd.Series | pd.DataFrame | dict[str, float] | str]:
    model_path = train_model(config)
    result = evaluate_saved_model(config, model_path=model_path)
    save_single_result(config, result)
    return result


def run_multiple_rl_baselines(
    configs: list[RLBaselineConfig],
) -> dict[str, dict[str, pd.Series | pd.DataFrame | dict[str, float] | str]]:
    results: dict[str, dict[str, pd.Series | pd.DataFrame | dict[str, float] | str]] = {}
    for config in configs:
        print(f"\n=== Training and evaluating {config.name} ===")
        results[config.name] = run_single_rl_baseline(config)

    plot_rl_equity_curves(results, output_path=FIGURES_DIR / "rl_baseline.png")
    save_aggregate_results(results, tables_dir=RL_TABLES_DIR)
    return results


def evaluate_multiple_saved_models(
    configs: list[RLBaselineConfig],
) -> dict[str, dict[str, pd.Series | pd.DataFrame | dict[str, float] | str]]:
    results: dict[str, dict[str, pd.Series | pd.DataFrame | dict[str, float] | str]] = {}
    for config in configs:
        model_path = config.model_dir / f"{config.name}.zip"
        print(f"\n=== Evaluating saved model {config.name} ===")
        result = evaluate_saved_model(config, model_path=model_path)
        save_single_result(config, result)
        results[config.name] = result

    plot_rl_equity_curves(results, output_path=FIGURES_DIR / "rl_baseline.png")
    save_aggregate_results(results, tables_dir=RL_TABLES_DIR)
    return results
