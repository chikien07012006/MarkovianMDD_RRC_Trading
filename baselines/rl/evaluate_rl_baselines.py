from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.rl.rl_baseline_common import evaluate_multiple_saved_models
from baselines.rl.train_ppo_markovian_mdd_static import build_config as build_ppo_markovian_mdd_static_config
from baselines.rl.train_ppo_profit_only import build_config as build_ppo_profit_only_config
from baselines.rl.train_ppo_variance_penalized import build_config as build_ppo_variance_penalized_config
from baselines.rl.train_sac_profit_only import build_config as build_sac_profit_only_config


def main() -> None:
    configs = [
        build_ppo_profit_only_config(),
        build_sac_profit_only_config(),
        build_ppo_variance_penalized_config(),
        build_ppo_markovian_mdd_static_config(),
    ]
    results = evaluate_multiple_saved_models(configs)
    for baseline_name, result in results.items():
        print(f"[{baseline_name}]")
        for metric_name, metric_value in result["metrics"].items():
            print(f"  {metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
