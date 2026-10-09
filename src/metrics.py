"""Evaluation metric for the Rossmann forecast.

The competition scores RMSPE (root mean squared percentage error). We use the
same metric everywhere (baselines, validation, error analysis) so every number
in the project is comparable to every other.
"""
import numpy as np


def rmspe(y_true, y_pred) -> float:
    """Root mean squared percentage error, ignoring days with zero sales.

    Why a *percentage* error: store sizes range from ~2,700 to ~21,000 daily
    sales (EDA section 4). An absolute error of 500 is a 19% miss for a small
    store but a 2% miss for a big one. RMSE would let the big stores dominate;
    RMSPE weights every store's relative accuracy equally.

    Why zeros are dropped: the percentage error is undefined when the true
    value is 0. Those are closed days, which we predict as 0 by rule anyway
    (EDA section 2), so excluding them loses nothing.

    Note the asymmetry: an under-prediction can cost at most 100% (predicting
    0), but an over-prediction is unbounded (predicting 3x the truth costs
    200%). The metric therefore rewards erring slightly low. That's why we
    train on log(Sales), which treats ratios symmetrically, and may later
    scale predictions down slightly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    mask = y_true != 0
    if not mask.any():
        raise ValueError("RMSPE is undefined when every true value is zero")

    pct_err = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(pct_err ** 2)))
