"""Combine multiple trained regressors by averaging their ``predict`` outputs."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """Mean pinball loss (quantile check loss) for a single quantile level.

    ``y_true`` and ``y_pred`` are broadcast-compatible (e.g. same shape).
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1)")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_true - y_pred
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def _index_quantile(levels: Sequence[float], q: float) -> Optional[int]:
    for i, v in enumerate(levels):
        if math.isclose(float(v), q, rel_tol=1e-9, abs_tol=1e-12):
            return i
    return None


class EnsembleRegressor:
    """Weighted or uniform average of models that expose ``predict(dataset, **kwargs)``.

    All members must share the same ``n_tasks``. Typical members are
    :class:`~models.gnn_regression.GNNRegressor`,
    :class:`~models.nn.pyg_regressor.PyGMoleculeRegressor`, or
    :class:`~models.hf_regression.HuggingFaceRegressor` trained on the same
    task layout (same ``n_tasks`` and compatible dataset type per model).

    ``predict`` forwards ``**kwargs`` to each member (e.g. ``batch_size``,
    ``show_progress``).
    """

    def __init__(
        self,
        models: Sequence[Any],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if not models:
            raise ValueError("models must be a non-empty sequence")
        self._models: Tuple[Any, ...] = tuple(models)
        n0 = int(getattr(self._models[0], "n_tasks", 0))
        if n0 < 1:
            raise ValueError("each model must have attribute n_tasks >= 1")
        for i, m in enumerate(self._models[1:], start=1):
            ni = int(getattr(m, "n_tasks", 0))
            if ni != n0:
                raise ValueError(
                    f"model[0] has n_tasks={n0}, model[{i}] has n_tasks={ni}"
                )
        self.n_tasks = n0

        if weights is None:
            w = np.ones(len(self._models), dtype=np.float64) / len(self._models)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != len(self._models):
                raise ValueError(
                    f"weights length {w.shape[0]} != number of models {len(self._models)}"
                )
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            s = float(w.sum())
            if s <= 0:
                raise ValueError("weights must sum to a positive value")
            w = w / s
        self._weights = w

    @property
    def models(self) -> Tuple[Any, ...]:
        return self._models

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    def fit(self, train_dataset: Any, **kwargs: Any) -> List[Any]:
        """Call ``fit`` on each member with the same ``train_dataset`` and ``kwargs``.

        Returns a list of per-model training outputs (typically loss histories).
        Members without ``fit`` raise ``TypeError``.
        """
        histories: List[Any] = []
        for m in self._models:
            fit = getattr(m, "fit", None)
            if not callable(fit):
                raise TypeError(
                    f"{type(m).__name__!r} has no callable fit; train members separately"
                )
            histories.append(fit(train_dataset, **kwargs))
        return histories

    def predict(self, dataset: Any, **kwargs: Any) -> np.ndarray:
        """Average predictions; shape ``(n_samples, n_tasks)``."""
        parts = [m.predict(dataset, **kwargs) for m in self._models]
        stacked = np.stack(parts, axis=0)
        w = self._weights.reshape(-1, 1, 1)
        out = np.sum(stacked * w, axis=0)
        return np.asarray(out, dtype=np.float64)

    def evaluate_loss(self, dataset: Any, **kwargs: Any) -> float:
        """Mean squared error between ensemble predictions and ``dataset.y``."""
        y = getattr(dataset, "y", None)
        if y is None:
            raise TypeError(
                "dataset must have a .y property (e.g. GraphRegressionDataset, "
                "SmilesRegressionDataset)"
            )
        pred = self.predict(dataset, **kwargs)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        return float(np.mean((pred - y_arr) ** 2))


class EnsembleQuantileRegressor:
    """Weighted average of quantile regressors that expose ``predict_quantiles``.

    Each member must implement ``predict_quantiles(dataset, **kwargs)`` returning
    ``float64`` array of shape ``(n_samples, n_tasks, n_quantiles)`` aligned with
    ``quantile_levels``. Typical use: several independently trained quantile heads
    or models (e.g. different seeds/architectures), combined like a prediction
    interval ensemble.

    Point prediction: :meth:`predict` returns the slice at the median quantile
    (``0.5``) when present in ``quantile_levels``; otherwise raises.
    """

    def __init__(
        self,
        models: Sequence[Any],
        quantile_levels: Sequence[float],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if not models:
            raise ValueError("models must be a non-empty sequence")
        if not quantile_levels:
            raise ValueError("quantile_levels must be non-empty")
        self._models: Tuple[Any, ...] = tuple(models)
        self.quantile_levels: Tuple[float, ...] = tuple(float(q) for q in quantile_levels)
        self._n_q = len(self.quantile_levels)

        n0 = int(getattr(self._models[0], "n_tasks", 0))
        if n0 < 1:
            raise ValueError("each model must have attribute n_tasks >= 1")
        for i, m in enumerate(self._models[1:], start=1):
            ni = int(getattr(m, "n_tasks", 0))
            if ni != n0:
                raise ValueError(
                    f"model[0] has n_tasks={n0}, model[{i}] has n_tasks={ni}"
                )
        self.n_tasks = n0

        if weights is None:
            w = np.ones(len(self._models), dtype=np.float64) / len(self._models)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != len(self._models):
                raise ValueError(
                    f"weights length {w.shape[0]} != number of models {len(self._models)}"
                )
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            s = float(w.sum())
            if s <= 0:
                raise ValueError("weights must sum to a positive value")
            w = w / s
        self._weights = w

    @property
    def models(self) -> Tuple[Any, ...]:
        return self._models

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    def fit(self, train_dataset: Any, **kwargs: Any) -> List[Any]:
        """Call ``fit`` on each member with the same ``train_dataset`` and ``kwargs``."""
        histories: List[Any] = []
        for m in self._models:
            fit = getattr(m, "fit", None)
            if not callable(fit):
                raise TypeError(
                    f"{type(m).__name__!r} has no callable fit; train members separately"
                )
            histories.append(fit(train_dataset, **kwargs))
        return histories

    def predict_quantiles(self, dataset: Any, **kwargs: Any) -> np.ndarray:
        """Weighted mean over members; shape ``(n_samples, n_tasks, n_quantiles)``."""
        parts = [m.predict_quantiles(dataset, **kwargs) for m in self._models]
        stacked = np.stack(parts, axis=0)
        if stacked.shape[-1] != self._n_q:
            raise ValueError(
                f"predict_quantiles last dim is {stacked.shape[-1]}, "
                f"expected {self._n_q} (len(quantile_levels))"
            )
        w = self._weights.reshape(-1, 1, 1, 1)
        out = np.sum(stacked * w, axis=0)
        return np.asarray(out, dtype=np.float64)

    def predict(self, dataset: Any, **kwargs: Any) -> np.ndarray:
        """Median quantile slice; shape ``(n_samples, n_tasks)``."""
        idx = _index_quantile(self.quantile_levels, 0.5)
        if idx is None:
            raise ValueError(
                "predict() requires a quantile level near 0.5; use predict_quantiles()"
            )
        q = self.predict_quantiles(dataset, **kwargs)
        return np.asarray(q[..., idx], dtype=np.float64)

    def evaluate_pinball_loss(self, dataset: Any, **kwargs: Any) -> float:
        """Mean pinball loss averaged over samples, tasks, and quantile levels."""
        y = getattr(dataset, "y", None)
        if y is None:
            raise TypeError(
                "dataset must have a .y property (e.g. GraphRegressionDataset, "
                "SmilesRegressionDataset)"
            )
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        pred = self.predict_quantiles(dataset, **kwargs)
        if pred.shape[:2] != y_arr.shape[:2]:
            raise ValueError(
                f"predict_quantiles shape {pred.shape[:2]} incompatible with y {y_arr.shape[:2]}"
            )
        losses: List[float] = []
        for j, q in enumerate(self.quantile_levels):
            losses.append(pinball_loss(y_arr, pred[..., j], q))
        return float(np.mean(losses))
