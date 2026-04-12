"""Phys-property gated mixture of Morgan fingerprint quantile ensembles."""

from __future__ import annotations

import math
import warnings
from typing import Any, List, Optional, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.ensemble import EnsembleQuantileRegressor, pinball_loss


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _row_normalize(x: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    a = np.maximum(np.asarray(x, dtype=np.float64), floor)
    return a / np.sum(a, axis=1, keepdims=True)


def _pop_dl_kwargs(kwargs: dict[str, Any]) -> None:
    for k in (
        "epochs",
        "batch_size",
        "show_progress",
        "val_dataset",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "lr_reduce_factor",
        "max_lr_reductions",
        "min_lr",
        "learning_rate",
        "weight_decay",
    ):
        kwargs.pop(k, None)


def _index_median(levels: Sequence[float]) -> Optional[int]:
    for i, v in enumerate(levels):
        if math.isclose(float(v), 0.5, rel_tol=1e-9, abs_tol=1e-12):
            return i
    return None


class FingerprintQuantileMember:
    """Fingerprint quantile regressors aligned to a dataset by row order.

    Fits one :class:`~sklearn.pipeline.Pipeline` per quantile level (HGB quantile
    loss by default). Use ``X_fp=`` for the feature matrix (e.g. Morgan counts).

    Only ``n_tasks == 1`` is supported (HistGradientBoosting quantile is
    single-target).
    """

    def __init__(
        self,
        model_name: str,
        estimator_template: Any,
        *,
        quantile_levels: Sequence[float],
        n_tasks: int = 1,
    ) -> None:
        if n_tasks != 1:
            raise ValueError("FingerprintQuantileMember only supports n_tasks=1")
        self.n_tasks = n_tasks
        self._model_name = model_name
        self._estimator_template = estimator_template
        self.quantile_levels: tuple[float, ...] = tuple(float(q) for q in quantile_levels)
        self._pipes: list[Any] = []

    def fit(self, dataset: Any, **kwargs: Any) -> List[float]:
        from baseline import make_regressor_pipeline

        X_fp = kwargs.pop("X_fp", None)
        if X_fp is None:
            raise TypeError(
                "FingerprintQuantileMember.fit(..., X_fp=) is required "
                "(one row per dataset sample, same order as dataset.y)"
            )
        sample_weight = kwargs.pop("sample_weight", None)
        _pop_dl_kwargs(kwargs)
        if kwargs:
            raise TypeError(
                f"FingerprintQuantileMember.fit got unexpected kwargs: {sorted(kwargs)!r}"
            )

        X_fp = np.asarray(X_fp, dtype=np.float64)
        y = np.asarray(getattr(dataset, "y"), dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X_fp.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_fp has {X_fp.shape[0]} rows, dataset.y has {y.shape[0]}"
            )
        if y.shape[1] != 1:
            raise ValueError("FingerprintQuantileMember expects a single target column")

        self._pipes = []
        fit_kw: dict[str, Any] = {}
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if sw.shape[0] != X_fp.shape[0]:
                raise ValueError(
                    f"sample_weight length {sw.shape[0]} != n_samples {X_fp.shape[0]}"
                )
            # Pipeline only routes weights to steps via step__param (final step is "model").
            fit_kw["model__sample_weight"] = sw

        for q in self.quantile_levels:
            est = clone(self._estimator_template)
            est.set_params(loss="quantile", quantile=q)
            pipe = make_regressor_pipeline(self._model_name, est)
            pipe.fit(X_fp, y.ravel(), **fit_kw)
            self._pipes.append(pipe)
        return []

    def predict_quantiles(self, dataset: Any, **kwargs: Any) -> np.ndarray:
        if len(self._pipes) != len(self.quantile_levels):
            raise RuntimeError("fit must be called before predict_quantiles")
        X_fp = kwargs.pop("X_fp", None)
        if X_fp is None:
            raise TypeError(
                "FingerprintQuantileMember.predict_quantiles(..., X_fp=) is required"
            )
        _pop_dl_kwargs(kwargs)
        kwargs.pop("sample_weight", None)
        if kwargs:
            raise TypeError(
                "FingerprintQuantileMember.predict_quantiles got unexpected kwargs: "
                f"{sorted(kwargs)!r}"
            )

        X_fp = np.asarray(X_fp, dtype=np.float64)
        y = np.asarray(getattr(dataset, "y"), dtype=np.float64)
        if X_fp.shape[0] != len(y):
            raise ValueError(
                f"X_fp has {X_fp.shape[0]} rows, dataset has {len(y)} samples"
            )
        n = X_fp.shape[0]
        n_q = len(self._pipes)
        out = np.empty((n, self.n_tasks, n_q), dtype=np.float64)
        for j, pipe in enumerate(self._pipes):
            p = np.asarray(pipe.predict(X_fp), dtype=np.float64).reshape(-1, 1)
            out[:, :, j] = p
        return out


class PhysGatedMorganQuantileMoE:
    """Mixture of quantile ensembles on Morgan counts with flexible routing.

    **Gated experts (K = ``n_components_gmm``):** trained with sample weights
    from a **Gaussian mixture** on three physicochemical descriptors (MolWt,
    MolLogP, TPSA), shape ``(n_samples, 3)``.

    **Optional PCA + GPR routing:** When ``pca_n_components_for_gpr`` is set and
    ``gmm_gpr_blend < 1``, a :class:`~sklearn.decomposition.PCA` of the Morgan
    matrix feeds a multi-output :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
    trained to match GMM responsibilities on the training set. At prediction,
    gated mixing weights are a convex blend of **GMM posteriors** and a
    **softmax of GPR outputs** on PCA scores (structure-informed routing).

    **Optional ungated experts:** ``n_ungated_experts`` additional
    :class:`~models.ensemble.EnsembleQuantileRegressor` stacks, each fit with
    uniform weights on all samples. They receive a fixed share ``ungated_mass``
    of the mixture (split equally), independent of phys / GPR routing.

    Expert features are typically ``morgan_r1_count_1024`` from
    :func:`features_data.build_descriptor_matrix`.
    """

    def __init__(
        self,
        *,
        n_components_gmm: int,
        quantile_levels: Sequence[float],
        n_ensemble_members: int = 3,
        weight_floor: float = 1e-6,
        model_name: str = "hgb",
        gmm_kwargs: Optional[dict[str, Any]] = None,
        hgb_kwargs: Optional[dict[str, Any]] = None,
        ensemble_weights: Optional[Sequence[float]] = None,
        random_state: Optional[int] = 0,
        pca_n_components_for_gpr: Optional[int] = None,
        gmm_gpr_blend: float = 0.5,
        n_ungated_experts: int = 0,
        ungated_mass: Optional[float] = None,
        gpr_max_train_samples: Optional[int] = 1_500,
        gpr_alpha: float = 0.05,
    ) -> None:
        if n_components_gmm < 1:
            raise ValueError("n_components_gmm must be >= 1")
        if n_ensemble_members < 1:
            raise ValueError("n_ensemble_members must be >= 1")
        if not quantile_levels:
            raise ValueError("quantile_levels must be non-empty")
        if not (0.0 <= gmm_gpr_blend <= 1.0):
            raise ValueError("gmm_gpr_blend must be in [0, 1]")
        if n_ungated_experts < 0:
            raise ValueError("n_ungated_experts must be >= 0")
        um: float
        if n_ungated_experts == 0:
            um = 0.0 if ungated_mass is None else float(ungated_mass)
            if um != 0.0:
                raise ValueError("ungated_mass must be 0 when n_ungated_experts is 0")
        else:
            if ungated_mass is None:
                um = 0.15
            else:
                um = float(ungated_mass)
            if not (0.0 < um < 1.0):
                raise ValueError("ungated_mass must be in (0, 1) when n_ungated_experts >= 1")
        if pca_n_components_for_gpr is not None:
            if int(pca_n_components_for_gpr) < 1:
                raise ValueError("pca_n_components_for_gpr must be >= 1 when set")
        self.n_components_gmm = n_components_gmm
        self.quantile_levels: tuple[float, ...] = tuple(float(q) for q in quantile_levels)
        self.n_ensemble_members = n_ensemble_members
        self.weight_floor = float(weight_floor)
        self._model_name = model_name
        self._gmm_kw = dict(gmm_kwargs or {})
        self._hgb_kw = dict(hgb_kwargs or {})
        self._ensemble_weights = ensemble_weights
        self.random_state = random_state
        self.pca_n_components_for_gpr = (
            int(pca_n_components_for_gpr)
            if pca_n_components_for_gpr is not None
            else None
        )
        self.gmm_gpr_blend = float(gmm_gpr_blend)
        self.n_ungated_experts = int(n_ungated_experts)
        self.ungated_mass = um
        self.gpr_max_train_samples = gpr_max_train_samples
        self.gpr_alpha = float(gpr_alpha)

        self.n_tasks = 1
        self._gate_pre: Optional[Pipeline] = None
        self._gmm: Optional[GaussianMixture] = None
        self._pca_morgan: Optional[PCA] = None
        self._gpr_router: Optional[MultiOutputRegressor] = None
        self._gated_experts: list[EnsembleQuantileRegressor] = []
        self._ungated_experts: list[EnsembleQuantileRegressor] = []

    def _build_expert(
        self,
        rs: int,
        expert_index: int,
    ) -> EnsembleQuantileRegressor:
        hgb_common: dict[str, Any] = {"max_iter": 200}
        hgb_common.update(self._hgb_kw)
        members: list[FingerprintQuantileMember] = []
        for j in range(self.n_ensemble_members):
            est = HistGradientBoostingRegressor(
                random_state=rs + expert_index * 1_000 + j * 17,
                **hgb_common,
            )
            members.append(
                FingerprintQuantileMember(
                    self._model_name,
                    est,
                    quantile_levels=self.quantile_levels,
                    n_tasks=1,
                )
            )
        return EnsembleQuantileRegressor(
            members,
            self.quantile_levels,
            weights=self._ensemble_weights,
        )

    def _gmm_posterior(self, X_gate: np.ndarray) -> np.ndarray:
        assert self._gate_pre is not None and self._gmm is not None
        Z = self._gate_pre.transform(X_gate)
        pi = self._gmm.predict_proba(Z)
        pi = np.maximum(pi, self.weight_floor)
        return _row_normalize(pi)

    def _gated_mixing_weights(self, X_gate: np.ndarray, X_morgan: np.ndarray) -> np.ndarray:
        """Rows sum to 1: per-sample weights over gated experts only."""
        pi_gmm = self._gmm_posterior(X_gate)
        if (
            self._pca_morgan is None
            or self._gpr_router is None
            or self.gmm_gpr_blend >= 1.0 - 1e-15
        ):
            return pi_gmm
        Zp = self._pca_morgan.transform(X_morgan)
        raw = self._gpr_router.predict(Zp)
        pi_gpr = _softmax_rows(raw)
        beta = self.gmm_gpr_blend
        mix = beta * pi_gmm + (1.0 - beta) * pi_gpr
        return _row_normalize(mix)

    def gated_mixing_weights(self, X_gate: np.ndarray, X_morgan: np.ndarray) -> np.ndarray:
        """Public alias for :meth:`_gated_mixing_weights` (for notebooks / callers)."""
        return self._gated_mixing_weights(X_gate, X_morgan)

    def fit(
        self,
        dataset: Any,
        *,
        X_gate: np.ndarray,
        X_morgan: np.ndarray,
        **kwargs: Any,
    ) -> List[Any]:
        _pop_dl_kwargs(kwargs)
        if kwargs:
            raise TypeError(
                f"PhysGatedMorganQuantileMoE.fit got unexpected kwargs: {sorted(kwargs)!r}"
            )

        X_gate = np.asarray(X_gate, dtype=np.float64)
        X_morgan = np.asarray(X_morgan, dtype=np.float64)
        if X_gate.ndim != 2 or X_gate.shape[1] != 3:
            raise ValueError("X_gate must have shape (n_samples, 3) for MolWt, MolLogP, TPSA")
        y = np.asarray(getattr(dataset, "y"), dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n = y.shape[0]
        if X_gate.shape[0] != n or X_morgan.shape[0] != n:
            raise ValueError("X_gate, X_morgan, and dataset.y must have the same n_samples")

        self._gate_pre = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
        Z = self._gate_pre.fit_transform(X_gate)
        rs = 0 if self.random_state is None else int(self.random_state)
        self._gmm = GaussianMixture(
            n_components=self.n_components_gmm,
            random_state=rs,
            **self._gmm_kw,
        )
        self._gmm.fit(Z)
        pi = self._gmm.predict_proba(Z)
        pi = np.maximum(pi, self.weight_floor)
        pi = _row_normalize(pi)

        self._pca_morgan = None
        self._gpr_router = None
        if self.pca_n_components_for_gpr is not None and self.gmm_gpr_blend < 1.0 - 1e-15:
            n_feat = X_morgan.shape[1]
            n_comp = min(self.pca_n_components_for_gpr, n, n_feat)
            if n_comp < 1:
                raise ValueError("PCA/GPR routing needs at least one component and sample")
            self._pca_morgan = PCA(n_components=n_comp, random_state=rs)
            Z_m = self._pca_morgan.fit_transform(X_morgan)
            n_fit = n
            idx = np.arange(n)
            max_tr = self.gpr_max_train_samples
            if max_tr is not None and n > int(max_tr):
                rng = np.random.default_rng(rs)
                idx = rng.choice(n, size=int(max_tr), replace=False)
                n_fit = int(max_tr)
            if n_fit < max(10, n_comp + 1):
                warnings.warn(
                    "GPR routing fit uses very few samples; consider disabling "
                    "pca_n_components_for_gpr or lowering n_components.",
                    UserWarning,
                    stacklevel=2,
                )
            kernel = (
                ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2))
                * RBF(
                    length_scale=np.ones(n_comp),
                    length_scale_bounds=(1e-2, 1e2),
                )
                + WhiteKernel(
                    noise_level=self.gpr_alpha,
                    noise_level_bounds=(1e-4, 1e2),
                )
            )
            base_gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=rs,
                alpha=0.0,
            )
            self._gpr_router = MultiOutputRegressor(base_gp, n_jobs=-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self._gpr_router.fit(Z_m[idx], pi[idx])

        self._gated_experts = []
        for k in range(self.n_components_gmm):
            expert = self._build_expert(rs, k)
            w = pi[:, k]
            for mem in expert.models:
                mem.fit(dataset, X_fp=X_morgan, sample_weight=w)
            self._gated_experts.append(expert)

        self._ungated_experts = []
        ones_w = np.ones(n, dtype=np.float64)
        for u in range(self.n_ungated_experts):
            expert = self._build_expert(rs, 10_000 + u)
            for mem in expert.models:
                mem.fit(dataset, X_fp=X_morgan, sample_weight=ones_w)
            self._ungated_experts.append(expert)

        return []

    def predict_quantiles(self, dataset: Any, **kwargs: Any) -> np.ndarray:
        if self._gate_pre is None or self._gmm is None or not self._gated_experts:
            raise RuntimeError("fit must be called before predict_quantiles")
        X_gate = kwargs.pop("X_gate", None)
        X_morgan = kwargs.pop("X_morgan", None)
        _pop_dl_kwargs(kwargs)
        if kwargs:
            raise TypeError(
                "PhysGatedMorganQuantileMoE.predict_quantiles got unexpected kwargs: "
                f"{sorted(kwargs)!r}"
            )
        if X_gate is None or X_morgan is None:
            raise TypeError("predict_quantiles requires X_gate= and X_morgan=")

        X_gate = np.asarray(X_gate, dtype=np.float64)
        X_morgan = np.asarray(X_morgan, dtype=np.float64)
        y = np.asarray(getattr(dataset, "y"), dtype=np.float64)
        n = len(y)
        if X_gate.shape != (n, 3):
            raise ValueError(f"X_gate must be (n, 3); got {X_gate.shape}, n={n}")
        if X_morgan.shape[0] != n:
            raise ValueError("X_morgan row count must match dataset length")

        pi_gate = self._gated_mixing_weights(X_gate, X_morgan)
        m_u = self.ungated_mass
        scale_g = 1.0 - m_u
        acc: Optional[np.ndarray] = None
        for k, expert in enumerate(self._gated_experts):
            qk = expert.predict_quantiles(dataset, X_fp=X_morgan)
            w = (scale_g * pi_gate[:, k]).reshape(-1, 1, 1)
            acc = qk * w if acc is None else acc + qk * w
        if self._ungated_experts:
            w_u = (m_u / len(self._ungated_experts)) if self._ungated_experts else 0.0
            for expert in self._ungated_experts:
                qk = expert.predict_quantiles(dataset, X_fp=X_morgan)
                w = np.full((n, 1, 1), w_u, dtype=np.float64)
                assert acc is not None
                acc = acc + qk * w
        assert acc is not None
        return np.asarray(acc, dtype=np.float64)

    def predict(self, dataset: Any, **kwargs: Any) -> np.ndarray:
        idx = _index_median(self.quantile_levels)
        if idx is None:
            raise ValueError(
                "predict() requires quantile level 0.5 in quantile_levels; "
                "use predict_quantiles()"
            )
        q = self.predict_quantiles(dataset, **kwargs)
        return np.asarray(q[..., idx], dtype=np.float64)

    def evaluate_pinball_loss(self, dataset: Any, **kwargs: Any) -> float:
        y = getattr(dataset, "y", None)
        if y is None:
            raise TypeError("dataset must have a .y property")
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        pred = self.predict_quantiles(dataset, **kwargs)
        if pred.shape[:2] != y_arr.shape[:2]:
            raise ValueError(
                f"predict_quantiles shape {pred.shape[:2]} incompatible with y {y_arr.shape[:2]}"
            )
        losses: list[float] = []
        for j, q in enumerate(self.quantile_levels):
            losses.append(pinball_loss(y_arr, pred[..., j], q))
        return float(np.mean(losses))
