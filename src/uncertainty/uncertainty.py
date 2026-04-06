"""
Uncertainty helpers for baseline regression.

Residual-based intervals from cross-validated predictions give **marginal**
coverage under an exchangeability assumption; they are not full conditional
calibration. For assay-linked columns, we only compare distributions, not
claim epistemic separation from assay noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def residual_quantile_offsets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> tuple[float, float]:
    """
    Quantiles of OOF residuals r = y − ŷ. Use with same-CV ŷ for intervals:
    [ŷ + q_low, ŷ + q_high] contains roughly (q_high − q_low) of the residual mass.
    """
    r = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.quantile(r, q_low)), float(np.quantile(r, q_high))


def prediction_intervals_from_residuals(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_pred_oof: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-sample intervals using residual quantiles from **OOF** predictions.

    `y_pred` can be in-sample or test predictions; offsets come from
    (y_true, y_pred_oof) on the training fold used to calibrate residuals.

    Returns (lower, upper) with the same shape as y_pred.
    """
    lo_off, hi_off = residual_quantile_offsets(y_true, y_pred_oof, q_low, q_high)
    yp = np.asarray(y_pred, dtype=float)
    return yp + lo_off, yp + hi_off


def fit_quantile_gradient_boosting(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha_low: float = 0.05,
    alpha_high: float = 0.95,
    random_state: int = 0,
    max_depth: int = 3,
    n_estimators: int = 150,
    learning_rate: float = 0.08,
) -> tuple[GradientBoostingRegressor, GradientBoostingRegressor]:
    """Two GBR models with quantile loss for lower/upper bounds."""
    common = dict(
        loss="quantile",
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    m_low = GradientBoostingRegressor(alpha=alpha_low, **common)
    m_high = GradientBoostingRegressor(alpha=alpha_high, **common)
    m_low.fit(X, y)
    m_high.fit(X, y)
    return m_low, m_high


def quantile_intervals_predict(
    model_low: GradientBoostingRegressor,
    model_high: GradientBoostingRegressor,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return model_low.predict(X), model_high.predict(X)


def compare_interval_width_to_assay(
    df: pd.DataFrame,
    model_lower: np.ndarray,
    model_upper: np.ndarray,
    *,
    std_col: str = "pEC50_std.error (-log10(molarity))",
    lower_col: str = "pEC50_ci.lower (-log10(molarity))",
    upper_col: str = "pEC50_ci.upper (-log10(molarity))",
) -> pd.DataFrame:
    """
    Align model interval width with assay std / CI width (sanity check).

    Rows with missing assay columns are dropped from the returned frame.
    """
    w_model = np.asarray(model_upper, dtype=float) - np.asarray(
        model_lower, dtype=float
    )
    out = df.copy()
    out["_model_width"] = w_model
    out["_assay_std"] = out[std_col]
    out["_assay_ci_width"] = out[upper_col] - out[lower_col]
    sub = out[["_model_width", "_assay_std", "_assay_ci_width"]].dropna()
    return sub


def assay_model_width_correlation(
    df: pd.DataFrame,
    model_lower: np.ndarray,
    model_upper: np.ndarray,
    **kwargs: Any,
) -> dict[str, float]:
    """Pearson r between model interval width and assay uncertainty columns."""
    sub = compare_interval_width_to_assay(
        df, model_lower, model_upper, **kwargs
    )
    if len(sub) < 3:
        return {"r_model_vs_std": float("nan"), "r_model_vs_ci_width": float("nan")}
    r_std = float(
        np.corrcoef(sub["_model_width"], sub["_assay_std"])[0, 1]
    )
    r_ci = float(
        np.corrcoef(sub["_model_width"], sub["_assay_ci_width"])[0, 1]
    )
    return {"r_model_vs_std": r_std, "r_model_vs_ci_width": r_ci}
