"""Δ-learning (delta learning): regress residuals vs a fixed baseline predictor.

Given a property :math:`y` and a baseline estimate :math:`\\hat{y}^{(0)}` (physics-based
score, docking, or a cheap ML model), train a model on the **residual**

``delta = y - y_baseline``

and form final predictions ``y_hat = y_baseline + delta_hat``. This often improves
sample efficiency when the baseline captures coarse trends and the ML layer
learns systematic corrections.

This module is **array-based** (same rows as ``baseline.py`` / fingerprint
pipelines). The baseline values must be supplied explicitly for each split
(out-of-fold baselines for training on deltas are left to the caller).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils.validation import check_array, check_is_fitted

__all__ = [
    "DeltaLearningRegressor",
    "combine_predictions",
    "compute_delta",
]


def compute_delta(y: np.ndarray, y_baseline: np.ndarray) -> np.ndarray:
    """Elementwise ``y - y_baseline`` (1d or 2d broadcast-safe)."""
    y = np.asarray(y, dtype=np.float64)
    y_baseline = np.asarray(y_baseline, dtype=np.float64)
    if y.shape != y_baseline.shape:
        raise ValueError(
            f"y shape {y.shape} and y_baseline shape {y_baseline.shape} must match"
        )
    return y - y_baseline


def combine_predictions(y_baseline: np.ndarray, delta_pred: np.ndarray) -> np.ndarray:
    """``y_baseline + delta_pred`` (same shapes)."""
    y_baseline = np.asarray(y_baseline, dtype=np.float64)
    delta_pred = np.asarray(delta_pred, dtype=np.float64)
    if y_baseline.shape != delta_pred.shape:
        raise ValueError(
            f"y_baseline shape {y_baseline.shape} != delta_pred shape {delta_pred.shape}"
        )
    return y_baseline + delta_pred


class DeltaLearningRegressor(BaseEstimator):
    """Sklearn-style regressor on residuals; ``predict`` needs baseline values per row.

    Inherits :class:`~sklearn.base.BaseEstimator` for ``get_params`` / ``clone``;
    :meth:`predict` is **not** the standard ``predict(X)`` signature (it requires
    ``y_baseline``), so this estimator is not fully pipeline-compatible without
    a thin wrapper.

    Parameters
    ----------
    estimator
        Unfitted sklearn regressor with ``fit(X, y)`` and ``predict(X)``,
        trained on ``delta = y - y_baseline`` from :meth:`fit`.

    Attributes
    ----------
    estimator_ : estimator
        The fitted residual model.
    n_features_in_ : int
        Number of features seen at fit time (if exposed by ``estimator_``).
    """

    def __init__(self, estimator: Optional[Any] = None) -> None:
        self.estimator = estimator

    def _get_estimator(self) -> Any:
        if self.estimator is not None:
            return self.estimator
        return HistGradientBoostingRegressor(max_iter=200, random_state=0)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_baseline: np.ndarray,
    ) -> "DeltaLearningRegressor":
        """Fit ``estimator_`` on ``delta = y - y_baseline``."""
        X = check_array(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y_baseline = np.asarray(y_baseline, dtype=np.float64)
        self._squeeze_outputs_ = bool(y.ndim == 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_baseline.ndim == 1:
            y_baseline = y_baseline.reshape(-1, 1)
        if y.shape[0] != X.shape[0] or y_baseline.shape[0] != X.shape[0]:
            raise ValueError("X, y, and y_baseline must have the same number of rows")
        if y.shape != y_baseline.shape:
            raise ValueError("y and y_baseline must have the same shape")
        delta = compute_delta(y, y_baseline)
        self.estimator_ = clone(self._get_estimator())
        if delta.shape[1] == 1:
            self.estimator_.fit(X, delta.ravel())
        else:
            self.estimator_.fit(X, delta)
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = int(self.estimator_.n_features_in_)  # type: ignore[attr-defined]
        return self

    def predict(self, X: np.ndarray, y_baseline: np.ndarray) -> np.ndarray:
        """Return ``y_baseline + estimator_.predict(X)``."""
        check_is_fitted(self, "estimator_")
        X = check_array(X, dtype=np.float64)
        y_baseline = np.asarray(y_baseline, dtype=np.float64)
        if y_baseline.ndim == 1:
            y_baseline = y_baseline.reshape(-1, 1)
        if y_baseline.shape[0] != X.shape[0]:
            raise ValueError("y_baseline must have one row per row of X")
        d = self.estimator_.predict(X)
        d = np.asarray(d, dtype=np.float64)
        if d.ndim == 1:
            d = d.reshape(-1, 1)
        if d.shape != y_baseline.shape:
            raise ValueError(
                f"residual model output shape {d.shape} != y_baseline shape {y_baseline.shape}"
            )
        out = combine_predictions(y_baseline, d)
        if getattr(self, "_squeeze_outputs_", False):
            return np.squeeze(out, axis=1)
        return out
