from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.rl_baseline_common import RLBaselineConfig, run_single_rl_baseline


def build_config() -> RLBaselineConfig:
    return RLBaselineConfig(
        name="ppo_variance_penalized",
        algorithm="ppo",
        reward_mode="variance_penalized",
        lambda_base=1000.0,
        reward_kwargs={"lambda_penalty": 1000.0},
        retrain_if_exists=True,
    )


def main() -> None:
    result = run_single_rl_baseline(build_config())
    print(f"[{result['baseline']}]")
    for metric_name, metric_value in result["metrics"].items():
        print(f"  {metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
