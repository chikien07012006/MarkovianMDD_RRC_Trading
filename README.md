# Markovian MDD Reward Shaping with Risk-Regime Conditioning for RL Trading

## Overview

This repository is a lightweight research prototype for **risk-aware reinforcement learning in trading**.
The project studies how reward design affects portfolio behavior when the agent trades a single risky asset
(`SPY`) against cash under changing market volatility regimes.

The main research idea is to compare several reward formulations under a common trading environment:

- simple return-based rewards
- variance-penalized rewards
- Differential Sharpe Ratio rewards
- Markovian drawdown-aware rewards
- drawdown-aware rewards with dynamic risk aversion via **Risk-Regime Conditioning (RRC)**

The current codebase is focused on the **data pipeline**, the **Gymnasium trading environment**, and a
**benchmark-ready reward API** that can later be plugged into PPO or other RL algorithms.

## Research Motivation

Many RL trading papers suffer from two practical issues:

1. **Non-Markovian drawdown penalties**

   Maximum Drawdown is often computed from episode-level path statistics that are not included in the state.
   This breaks the Markov assumption and makes the reward harder to reason about theoretically.

2. **Static risk aversion**

   A fixed drawdown penalty coefficient does not adapt when markets shift from calm to stressed regimes.

This project addresses both issues by:

- augmenting the state with `running_peak`
- computing drawdown directly from stateful portfolio information
- adjusting the drawdown penalty dynamically using a VIX-based regime signal

## Current Scope

The repository currently includes:

- a daily SPY + VIX data ingestion pipeline
- indicator preprocessing for train/validation/test splits
- a `Gymnasium` trading environment with continuous actions
- four benchmark reward functions with a shared callable interface
- reward injection support for controlled PPO benchmarking later

The repository does **not yet** include:

- PPO training scripts
- Stable-Baselines3 integration
- experiment tracking
- baseline training/evaluation loops
- result tables or paper-ready figures

## Repository Structure

```text
MarkovianMDD_RRC_Trading/
|-- data/
|   |-- crawl_data.py
|   |-- preprocess_indicator_signals.py
|   |-- raw/
|   `-- processed/
|-- env/
|   `-- trading_env.py
|-- reward/
|   |-- __init__.py
|   |-- variance_penalized.py
|   |-- differential_sharpe.py
|   |-- markovian_mdd_static.py
|   |-- markovian_mdd.py
|   `-- rrc.py
|-- results/
|   |-- figures/
|   `-- tables/
|-- tests/
|-- requirements.txt
`-- README.md
```

## Trading Problem Setup

### Asset Universe

- **Primary asset:** `SPY`
- **Risk signal:** `^VIX`
- **Portfolio:** `SPY` and cash

### Action Space

- Continuous action `w in [-1, 1]`
- `w` is interpreted as the target portfolio weight on the risky asset

### Observation Space

The environment observation contains:

- market features
- portfolio cash ratio
- current risky-asset weight
- unrealized PnL
- `running_peak`

Default market features:

- `log_return`
- `sma_ratio`
- `rsi_14`
- `bollinger_band_width`
- `vix_zscore_252`

### Portfolio Dynamics

The environment currently models:

- next-step price transition using daily close-to-close returns
- target-weight reallocation
- turnover-based transaction cost
- NAV evolution
- running peak tracking
- Markovian drawdown tracking

### Transaction Cost

- `0.1%` per unit turnover via `transaction_cost_rate=0.001`

## Data

### Time Splits

- **Train:** `2010-01-01` to `2017-12-31`
- **Validation:** `2018-01-01` to `2019-12-31`
- **Test:** `2020-01-01` to `2022-12-31`

The test window includes both the **2020 COVID crash** and the **2022 bear market**.

### Raw Data Pipeline

Script: [data/crawl_data.py](data/crawl_data.py)

What it does:

- downloads daily `SPY` OHLCV data from `yfinance`
- downloads daily `^VIX` close data
- aligns both series by date
- writes split CSV files into `data/raw`

Generated files:

```text
data/raw/spy_vix_train.csv
data/raw/spy_vix_validation.csv
data/raw/spy_vix_test.csv
```

### Feature Engineering Pipeline

Script: [data/preprocess_indicator_signals.py](data/preprocess_indicator_signals.py)

Computed features:

- `log_return`
- `sma_ratio`
- `rsi_14`
- `bollinger_band_width`
- `vix_zscore_252`

Generated files:

```text
data/processed/spy_vix_indicators_train.csv
data/processed/spy_vix_indicators_validation.csv
data/processed/spy_vix_indicators_test.csv
```

Note:

- the train split can contain warmup `NaN` values from rolling indicators
- you can remove these rows with the `--drop-na` option during preprocessing

## Reward Library

All reward functions live in `reward/` and follow a common callable pattern:

```python
reward = reward_fn(env, transition)
```

This makes them easy to inject into the environment for later PPO benchmarking.

### R0: Log Return Baseline

Default environment reward:

```text
reward_t = portfolio_log_return_t
```

This is used when `reward_mode="default"` or when no custom reward function is injected.

### R1: Variance-Penalized Return

File: [reward/variance_penalized.py](reward/variance_penalized.py)

Formula:

```text
reward_t = log_return_t - lambda * portfolio_variance_t
```

Current implementation details:

- default `lambda = 1.0`
- variance is tracked online within each episode
- reward state is reset automatically at `env.reset()`

### R2: Differential Sharpe Ratio

File: [reward/differential_sharpe.py](reward/differential_sharpe.py)

This reward uses EMA statistics of returns and computes a differential Sharpe-style signal inspired by
Moody and Saffell (2001).

Current implementation details:

- default EMA step size `eta = 0.01`
- online mean and second-moment updates
- reward state is reset automatically at `env.reset()`

### R3: Markovian MDD Reward with Static Lambda

File: [reward/markovian_mdd_static.py](reward/markovian_mdd_static.py)

Formula:

```text
drawdown_t = max(0, running_peak_t - NAV_t) / running_peak_t
reward_t = log_return_t - lambda * drawdown_t
```

Current implementation details:

- default `lambda = 1.0`
- uses `running_peak` maintained by the environment
- keeps drawdown penalty fully compatible with the Markov state design

### R4: Markovian MDD + Risk-Regime Conditioning

Files:

- [reward/markovian_mdd.py](reward/markovian_mdd.py)
- [reward/rrc.py](reward/rrc.py)

Formula:

```text
lambda_rrc(t) = lambda_base * (1 + alpha * clamp(vix_zscore_t, -3, 3))
reward_t = log_return_t - lambda_rrc(t) * drawdown_t
```

Current implementation details:

- default `lambda_base = 1.0`
- default `alpha = 0.0`
- VIX z-score is clipped to `[-3, 3]`
- resulting lambda is floored at `0.0`

## Trading Environment

Main file: [env/trading_env.py](env/trading_env.py)

The environment is designed for reward benchmarking and exposes a rich `transition` dictionary to the
reward function on every step.

Key transition fields include:

- `portfolio_log_return`
- `drawdown`
- `nav`
- `previous_nav`
- `running_peak`
- `turnover`
- `transaction_cost`
- `vix_zscore_t`
- `market_features_t`

### Reward Injection Modes

You can use the built-in reward selector:

- `default` or `r0`
- `variance_penalized` or `r1`
- `differential_sharpe` or `r2`
- `markovian_mdd` or `r3`
- `markovian_mdd_rrc` or `r4`

Or you can inject a reward object directly through `reward_fn`.

### Example: Built-In Reward Modes

```python
from env.trading_env import TradingEnv

env = TradingEnv(
    data="data/processed/spy_vix_indicators_validation.csv",
    reward_mode="r4",
    lambda_base=1.0,
    alpha=0.2,
)
```

### Example: Direct Reward Injection

```python
from env.trading_env import TradingEnv
from reward import build_differential_sharpe_reward

env = TradingEnv(
    data="data/processed/spy_vix_indicators_validation.csv",
    reward_fn=build_differential_sharpe_reward(eta=0.05),
)
```

### Example: Reward-Specific Keyword Arguments

```python
from env.trading_env import TradingEnv

env = TradingEnv(
    data="data/processed/spy_vix_indicators_validation.csv",
    reward_mode="r2",
    reward_kwargs={"eta": 0.05, "epsilon": 1e-12},
)
```

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Current dependencies:

- `pandas`
- `numpy`
- `yfinance`
- `gymnasium`

## Data Preparation

Download and split raw data:

```powershell
.\.venv\Scripts\python data\crawl_data.py
```

Preprocess features:

```powershell
.\.venv\Scripts\python data\preprocess_indicator_signals.py
```

Drop rolling-window warmup rows if needed:

```powershell
.\.venv\Scripts\python data\preprocess_indicator_signals.py --drop-na
```

## Current Project Status

### Implemented

- SPY + VIX data ingestion
- feature preprocessing and split generation
- Gymnasium trading environment
- reward benchmark library with 4 research reward formulations
- reward-state reset support for stateful rewards
- reward injection API for later PPO experiments

### Not Yet Implemented

- PPO training pipeline
- Stable-Baselines3 dependency integration
- benchmark runner across reward functions
- financial baselines such as Buy & Hold, Risk Parity, or CPPI
- evaluation scripts, plots, and result tables
- automated tests

## Recommended Next Steps

1. Add `stable-baselines3` and a PPO training script.
2. Build a benchmark runner that loops over `r0` to `r4`.
3. Add validation/test evaluation metrics such as return, volatility, Sharpe, max drawdown, and turnover.
4. Export experiment outputs into `results/tables` and `results/figures`.
5. Add reproducible configs and tests for reward correctness.

## Project Goal

The broader goal is to build a compact RL trading research framework that is:

- theoretically cleaner than episode-level drawdown penalties
- flexible enough to compare multiple reward designs fairly
- practical for constrained academic compute budgets
- extendable toward PPO-based reward benchmarking and regime-aware evaluation
