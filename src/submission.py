"""
Activity-track submission helpers for the PXR blind challenge.

Schema and merge semantics align with OpenADMET PXR-Challenge-Tutorial
``evaluation/evaluate_predictions.py`` and ``evaluation/config.py`` (MAE, RAE,
R2, Spearman, Kendall; optional bootstrap).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

# Mirrors evaluation/config.py
ACTIVITY_ENDPOINTS: tuple[str, ...] = ("pEC50",)
ENDPOINTS_TO_LOG_TRANSFORM: frozenset[str] = frozenset()
BOOTSTRAP_SAMPLES_DEFAULT: int = 1000


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative absolute error (same definition as tutorial config)."""
    denom = np.sum(np.abs(y_true - np.mean(y_true)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _spearman_stat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = spearmanr(y_true, y_pred)
    return float(r.statistic)


def _kendall_stat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = kendalltau(y_true, y_pred)
    return float(r.statistic)


ACTIVITY_METRICS: tuple[tuple[str, Callable[..., Any]], ...] = (
    ("MAE", mean_absolute_error),
    ("RAE", rae),
    ("R2", r2_score),
    ("Spearman R", _spearman_stat),
    ("Kendall's Tau", _kendall_stat),
)


def _bootstrap_indices(n: int, n_bootstrap: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    for _ in range(n_bootstrap):
        yield rng.integers(0, n, size=n)


def build_activity_submission(
    test_df: pd.DataFrame,
    predictions: np.ndarray | pd.Series,
    *,
    id_col: str = "Molecule Name",
    value_col: str = "pEC50",
) -> pd.DataFrame:
    """
    Build a submission table: one row per test compound with predicted ``value_col``.

    If ``predictions`` is a 1-D array, it must have length ``len(test_df)`` and is
    aligned to ``test_df`` row order. If it is a ``Series``, it must be indexed by
    ``id_col`` (molecule id) and include every id in ``test_df[id_col]``.
    """
    if id_col not in test_df.columns:
        raise KeyError(f"test_df missing column {id_col!r}")

    if isinstance(predictions, pd.Series):
        ids = test_df[id_col].tolist()
        s = predictions.reindex(ids)
        if s.isna().any():
            missing = [i for i, v in zip(ids, s, strict=True) if pd.isna(v)]
            raise ValueError(
                f"Missing predictions for {len(missing)} molecule(s), e.g. {missing[:5]!r}"
            )
        vals = s.to_numpy(dtype=float)
    else:
        vals = np.asarray(predictions, dtype=float).reshape(-1)
        if vals.shape[0] != len(test_df):
            raise ValueError(
                f"predictions length {vals.shape[0]} != len(test_df) {len(test_df)}"
            )

    out = pd.DataFrame({id_col: test_df[id_col].values, value_col: vals})
    return out


def validate_submission(
    sub: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    id_col: str = "Molecule Name",
    value_col: str = "pEC50",
) -> None:
    """
    Validate submission against the blind test table (expected ids, no NaNs).

    Raises ``ValueError`` with an actionable message if checks fail.
    """
    for col in (id_col, value_col):
        if col not in sub.columns:
            raise ValueError(f"Submission missing required column {col!r}")

    if id_col not in test_df.columns:
        raise KeyError(f"test_df missing column {id_col!r}")

    if len(sub) != len(test_df):
        raise ValueError(f"Submission row count {len(sub)} != test row count {len(test_df)}")

    sub_ids = sub[id_col].astype(str).values
    test_ids = test_df[id_col].astype(str).values
    if np.sort(sub_ids).tolist() != np.sort(test_ids).tolist():
        raise ValueError(
            "Submission Molecule Name multiset does not match test set "
            "(duplicate or missing IDs)."
        )

    if sub[value_col].isna().any():
        raise ValueError(f"Submission column {value_col!r} contains NaN.")

    if not np.issubdtype(sub[value_col].dtype, np.number):
        try:
            sub[value_col].astype(float)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Submission column {value_col!r} is not numeric.") from e


def write_submission(
    path: Path | str,
    sub: pd.DataFrame,
    *,
    format: Literal["csv", "parquet"] = "csv",
) -> None:
    """Write submission artifact (CSV or Parquet)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if format == "csv":
        sub.to_csv(p, index=False)
    else:
        sub.to_parquet(p, index=False)


def merge_predictions_with_truth(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    id_col: str = "Molecule Name",
) -> pd.DataFrame:
    """Merge like ``evaluate_predictions.score_activity_predictions`` (suffixes _pred / _true)."""
    return predictions.merge(
        ground_truth,
        on=id_col,
        suffixes=("_pred", "_true"),
        how="right",
    ).sort_values(id_col)


def score_activity_predictions_simple(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    endpoint: str = "pEC50",
    id_col: str = "Molecule Name",
) -> pd.Series:
    """
    Non-bootstrap metrics on the merged dataset (one value per metric name).

    Raises if merge yields NaNs in predicted column (matches strict tutorial behavior).
    """
    merged = merge_predictions_with_truth(predictions, ground_truth, id_col=id_col)
    col_pred = f"{endpoint}_pred"
    col_true = f"{endpoint}_true"
    if merged[col_pred].isna().any() or merged[col_true].isna().any():
        raise ValueError(
            "Merged predictions and ground truth contain NaN — missing predictions for some molecules."
        )
    y_pred = merged[col_pred].to_numpy(dtype=float)
    y_true = merged[col_true].to_numpy(dtype=float)

    out: dict[str, float] = {}
    for name, fn in ACTIVITY_METRICS:
        try:
            v = fn(y_true, y_pred)
        except Exception:
            v = float("nan")
        if not isinstance(v, (int, float)):
            v = float(v)
        out[name] = float(v)
    return pd.Series(out, name=endpoint)


def bootstrap_activity_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    endpoint: str = "pEC50",
    n_bootstrap_samples: int = BOOTSTRAP_SAMPLES_DEFAULT,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """
    Bootstrap rows with replacement; one row per iteration with metric columns.

    Same long-form shape as tutorial ``bootstrap_metrics`` (columns Sample, Endpoint,
    plus metric names).
    """
    n = y_true.shape[0]
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    for b, idx in enumerate(_bootstrap_indices(n, n_bootstrap_samples, rng)):
        row: dict[str, Any] = {"Sample": b, "Endpoint": endpoint}
        y_p = y_pred[idx]
        y_t = y_true[idx]
        for metric_name, metric_func in ACTIVITY_METRICS:
            try:
                val = metric_func(y_t, y_p)
            except Exception:
                val = np.nan
            if not isinstance(val, (int, float)):
                val = getattr(val, "statistic", val)
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = np.nan
            row[metric_name] = val
        rows.append(row)
    return pd.DataFrame(rows)


def average_bootstrap_activity_results(
    bootstrap_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mean and std of each metric column by Endpoint (tutorial-style aggregate)."""
    metric_cols = [c for c in bootstrap_df.columns if c not in ("Sample", "Endpoint")]
    if not metric_cols:
        return pd.DataFrame()
    g = bootstrap_df.drop(columns=["Sample"]).groupby("Endpoint", as_index=True)
    agg = g.agg(["mean", "std"])
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    return agg


def score_activity_with_bootstrap(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    endpoint: str = "pEC50",
    id_col: str = "Molecule Name",
    n_bootstrap_samples: int = BOOTSTRAP_SAMPLES_DEFAULT,
    random_state: int | None = 0,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Merge pred/truth, return (simple metrics, bootstrap long-form, bootstrap aggregate).
    """
    merged = merge_predictions_with_truth(predictions, ground_truth, id_col=id_col)
    col_pred = f"{endpoint}_pred"
    col_true = f"{endpoint}_true"
    if merged[col_pred].isna().any() or merged[col_true].isna().any():
        raise ValueError(
            "Merged predictions and ground truth contain NaN — missing predictions for some molecules."
        )
    y_pred = merged[col_pred].to_numpy(dtype=float)
    y_true = merged[col_true].to_numpy(dtype=float)
    simple = score_activity_predictions_simple(predictions, ground_truth, endpoint=endpoint, id_col=id_col)
    boot = bootstrap_activity_metrics(
        y_pred,
        y_true,
        endpoint=endpoint,
        n_bootstrap_samples=n_bootstrap_samples,
        random_state=random_state,
    )
    agg = average_bootstrap_activity_results(boot)
    return simple, boot, agg
