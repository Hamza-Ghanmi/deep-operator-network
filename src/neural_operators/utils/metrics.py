"""Evaluation metrics for neural operator predictions."""

from __future__ import annotations

import numpy as np

__all__ = ["mse", "relative_l2", "metrics_summary"]


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error over all elements."""
    return float(np.mean((pred - true) ** 2))


def relative_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample relative L2 error.

    Args:
        pred: (N, ...) predicted values.
        true: (N, ...) ground-truth values.

    Returns:
        (N,) array of relative L2 errors.
    """
    flat_p = pred.reshape(len(pred), -1)
    flat_t = true.reshape(len(true), -1)
    return np.linalg.norm(flat_p - flat_t, axis=1) / (np.linalg.norm(flat_t, axis=1) + 1e-8)


def metrics_summary(
    pred: np.ndarray,
    true: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    """Compute a standard set of evaluation metrics.

    Returns:
        ``(mse, rel_l2_mean, rel_l2_std, max_abs_error, rel_l2_per_sample)``
    """
    _mse = mse(pred, true)
    rl   = relative_l2(pred, true)
    return _mse, float(rl.mean()), float(rl.std()), float(np.abs(pred - true).max()), rl