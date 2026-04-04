from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_target_distribution(
    df: pd.DataFrame,
    col: str = "pEC50",
    *,
    ax: plt.Axes | None = None,
    bins: int = 40,
    **kwargs: Any,
) -> plt.Axes:
    """Histogram of the regression target."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    s = df[col].dropna()
    ax.hist(s, bins=bins, density=True, alpha=0.75, edgecolor="black", **kwargs)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {col}")
    return ax


def plot_assay_uncertainty_vs_target(
    df: pd.DataFrame,
    target_col: str = "pEC50",
    std_col: str = "pEC50_std.error (-log10(molarity))",
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter of assay-reported std error vs pEC50 (EDA)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sub = df[[target_col, std_col]].dropna()
    ax.scatter(sub[target_col], sub[std_col], alpha=0.25, s=8)
    ax.set_xlabel(target_col)
    ax.set_ylabel(std_col)
    ax.set_title("Assay uncertainty vs target")
    return ax


def plot_ci_width_vs_target(
    df: pd.DataFrame,
    target_col: str = "pEC50",
    lower_col: str = "pEC50_ci.lower (-log10(molarity))",
    upper_col: str = "pEC50_ci.upper (-log10(molarity))",
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    width = df[upper_col] - df[lower_col]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sub = pd.DataFrame({target_col: df[target_col], "width": width}).dropna()
    ax.scatter(sub[target_col], sub["width"], alpha=0.25, s=8)
    ax.set_xlabel(target_col)
    ax.set_ylabel("CI width")
    ax.set_title("Assay CI width vs target")
    return ax


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """CV predictions: y vs ŷ with identity line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, label=label)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted (CV OOF)")
    ax.set_title("Predicted vs actual")
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_residuals_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    res = y_true - y_pred
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, res, alpha=0.3, s=10)
    ax.axhline(0.0, color="k", linestyle="--", lw=1)
    ax.set_xlabel("Predicted (CV OOF)")
    ax.set_ylabel("Residual (y − ŷ)")
    ax.set_title("Residuals vs predicted")
    return ax


def plot_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    bins: int = 40,
) -> plt.Axes:
    res = y_true - y_pred
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(res, bins=bins, density=True, alpha=0.75, edgecolor="black")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution (CV OOF)")
    return ax


def plot_model_comparison(
    results_df: pd.DataFrame,
    *,
    metric: str = "mean_rmse",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Axes:
    """Bar chart of CV metric by descriptor × model (wide table)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    pivot = results_df.pivot_table(
        index="descriptor", columns="model", values=metric, aggfunc="first"
    )
    pivot.plot(kind="bar", ax=ax, rot=45)
    ax.set_ylabel(metric)
    ax.set_title(f"Baseline CV comparison ({metric})")
    ax.legend(title="model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel("descriptor")
    plt.tight_layout()
    return ax
