from __future__ import annotations


def compute_lambda_rrc(
    vix_zscore_t: float,
    lambda_base: float,
    alpha: float,
    clamp_min: float = -3.0,
    clamp_max: float = 3.0,
) -> float:
    if lambda_base < 0:
        raise ValueError("lambda_base must be non-negative.")
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if clamp_min > clamp_max:
        raise ValueError("clamp_min must be less than or equal to clamp_max.")

    clamped_vix = min(max(float(vix_zscore_t), clamp_min), clamp_max)
    lambda_rrc = lambda_base * (1.0 + (alpha * clamped_vix))
    return max(0.0, lambda_rrc)
