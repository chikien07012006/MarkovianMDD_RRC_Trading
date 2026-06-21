from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.ppo_hybrid_regime_aware_policy.pipeline import run_proposed_method_single_config


def main() -> None:
    result = run_proposed_method_single_config(
        total_timesteps=30_000,
        seed=42,
        lambda_base=0.05,
        alpha=0.50,
        beta_target=0.02,
    )
    config = result["config"]
    test_metrics = result["proposed_test_result"]["metrics"]

    print(
        f"\nSingle-run config: lambda_base={config.lambda_base:.2f}, "
        f"alpha={config.alpha:.2f}, "
        f"beta_target={config.reward_kwargs.get('beta_target', 0.0):.4f}"
    )
    print("[ppo_hybrid_regime_aware_policy]")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")


if __name__ == "__main__":
    main()
