"""Combine multiple trained regressors by averaging their ``predict`` outputs."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone


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


class FingerprintEnsembleMember:
    """Sklearn fingerprint pipeline aligned to a graph/SMILES dataset by row order.

    Use with :class:`EnsembleRegressor` together with GNN/HF members: pass the same
    ``GraphRegressionDataset`` (or any dataset with ``.y``) and provide matching
    ``X_fp`` rows to :meth:`fit` / :meth:`predict`.

    Parameters mirror ``baseline.make_regressor_pipeline(model_name, estimator)``.
    """

    def __init__(self, model_name: str, estimator: Any, *, n_tasks: int = 1) -> None:
        if n_tasks < 1:
            raise ValueError("n_tasks must be >= 1")
        self.n_tasks = n_tasks
        self._model_name = model_name
        self._estimator = estimator
        self._pipe: Any = None

    def fit(self, dataset: Any, **kwargs: Any) -> List[float]:
        from baseline import make_regressor_pipeline

        X_fp = kwargs.pop("X_fp", None)
        if X_fp is None:
            raise TypeError(
                "FingerprintEnsembleMember.fit(..., X_fp=) is required "
                "(one row per dataset sample, same order as dataset.y)"
            )
        kwargs.pop("epochs", None)
        kwargs.pop("batch_size", None)
        kwargs.pop("show_progress", None)
        kwargs.pop("val_dataset", None)
        kwargs.pop("early_stopping_patience", None)
        kwargs.pop("early_stopping_min_delta", None)
        kwargs.pop("lr_reduce_factor", None)
        kwargs.pop("max_lr_reductions", None)
        kwargs.pop("min_lr", None)
        kwargs.pop("learning_rate", None)
        kwargs.pop("weight_decay", None)
        X_fp = np.asarray(X_fp, dtype=np.float64)
        y = np.asarray(getattr(dataset, "y"), dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X_fp.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_fp has {X_fp.shape[0]} rows, dataset.y has {y.shape[0]}"
            )
        if y.shape[1] != self.n_tasks:
            raise ValueError(
                f"dataset has n_tasks={y.shape[1]}, member expects {self.n_tasks}"
            )
        self._pipe = make_regressor_pipeline(self._model_name, clone(self._estimator))
        if self.n_tasks == 1:
            self._pipe.fit(X_fp, y.ravel())
        else:
            self._pipe.fit(X_fp, y)
        return []

    def predict(self, dataset: Any, **kwargs: Any) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("fit must be called before predict")
        X_fp = kwargs.pop("X_fp", None)
        if X_fp is None:
            raise TypeError("FingerprintEnsembleMember.predict(..., X_fp=) is required")
        kwargs.pop("batch_size", None)
        kwargs.pop("show_progress", None)
        X_fp = np.asarray(X_fp, dtype=np.float64)
        y = np.asarray(getattr(dataset, "y"), dtype=np.float64)
        if X_fp.shape[0] != len(y):
            raise ValueError(
                f"X_fp has {X_fp.shape[0]} rows, dataset has {len(y)} samples"
            )
        p = self._pipe.predict(X_fp)
        p = np.asarray(p, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        return p


def _any_fingerprint_member(models: Sequence[Any]) -> bool:
    return any(isinstance(m, FingerprintEnsembleMember) for m in models)


class EnsembleRegressor:
    """Weighted or uniform average of models that expose ``predict(dataset, **kwargs)``.

    All members must share the same ``n_tasks``. Typical members are
    :class:`~models.gnn_regression.GNNRegressor` (default GAT; set ``architecture=``),
    :class:`~models.nn.pyg_regressor.PyGMoleculeRegressor`,
    :class:`~models.hf_regression.HuggingFaceRegressor`, or
    :class:`FingerprintEnsembleMember` (sklearn tree/GBDT on fingerprints).

    If any member is :class:`FingerprintEnsembleMember`, you must pass
    ``X_fp`` (feature matrix, same row order as ``dataset``) to :meth:`fit`,
    :meth:`predict`, and :meth:`evaluate_loss`.

    ``predict`` forwards ``**kwargs`` to each member (e.g. ``batch_size``,
    ``show_progress`` for DL models).
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

        When the ensemble includes :class:`FingerprintEnsembleMember`, pass
        ``X_fp`` aligned with ``train_dataset`` rows.
        """
        fp_kw: dict[str, Any] = {}
        if _any_fingerprint_member(self._models):
            X_fp = kwargs.pop("X_fp", None)
            if X_fp is None:
                raise ValueError(
                    "EnsembleRegressor.fit requires X_fp=... when a "
                    "FingerprintEnsembleMember is included"
                )
            fp_kw["X_fp"] = X_fp
        histories: List[Any] = []
        for m in self._models:
            fit = getattr(m, "fit", None)
            if not callable(fit):
                raise TypeError(
                    f"{type(m).__name__!r} has no callable fit; train members separately"
                )
            if isinstance(m, FingerprintEnsembleMember):
                histories.append(fit(train_dataset, **fp_kw, **kwargs))
            else:
                histories.append(fit(train_dataset, **kwargs))
        return histories

    def predict(self, dataset: Any, **kwargs: Any) -> np.ndarray:
        """Average predictions; shape ``(n_samples, n_tasks)``."""
        X_fp = kwargs.get("X_fp")
        if _any_fingerprint_member(self._models) and X_fp is None:
            raise ValueError(
                "EnsembleRegressor.predict requires X_fp=... when a "
                "FingerprintEnsembleMember is included"
            )
        parts: List[np.ndarray] = []
        dl_kw = {k: v for k, v in kwargs.items() if k != "X_fp"}
        for m in self._models:
            if isinstance(m, FingerprintEnsembleMember):
                parts.append(m.predict(dataset, **kwargs))
            else:
                parts.append(m.predict(dataset, **dl_kw))
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
