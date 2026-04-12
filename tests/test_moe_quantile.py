"""Tests for phys-gated Morgan quantile MoE."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from models.moe_quantile import FingerprintQuantileMember, PhysGatedMorganQuantileMoE


class _DS:
    def __init__(self, y: np.ndarray) -> None:
        self.y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.y)


def test_fingerprint_quantile_member_shapes_and_median():
    rng = np.random.default_rng(0)
    n, d = 40, 12
    X = rng.standard_normal((n, d))
    y = X[:, 0] + 0.1 * rng.standard_normal(n)
    ds = _DS(y)
    levels = [0.1, 0.5, 0.9]
    est = HistGradientBoostingRegressor(max_iter=30, max_depth=3, random_state=0)
    m = FingerprintQuantileMember("hgb", est, quantile_levels=levels, n_tasks=1)
    m.fit(ds, X_fp=X)
    q = m.predict_quantiles(ds, X_fp=X)
    assert q.shape == (n, 1, 3)
    assert np.all(q[..., 0] <= q[..., 2] + 1e-6)


def test_fingerprint_quantile_member_sample_weight():
    rng = np.random.default_rng(1)
    n, d = 30, 8
    X = rng.standard_normal((n, d))
    y = X[:, 0]
    ds = _DS(y)
    w = rng.random(n) + 0.1
    est = HistGradientBoostingRegressor(max_iter=20, max_depth=2, random_state=1)
    m = FingerprintQuantileMember("hgb", est, quantile_levels=[0.5], n_tasks=1)
    m.fit(ds, X_fp=X, sample_weight=w)
    q = m.predict_quantiles(ds, X_fp=X)
    assert q.shape == (n, 1, 1)


def test_phys_gated_moe_proba_and_quantile_shape():
    rng = np.random.default_rng(2)
    n = 120
    # Two blobs in gate space
    gate = np.vstack(
        [
            rng.standard_normal((n // 2, 3)) + np.array([3.0, 0.0, 0.0]),
            rng.standard_normal((n - n // 2, 3)) + np.array([-3.0, 0.0, 0.0]),
        ]
    )
    X_m = rng.standard_normal((n, 16))
    y = gate[:, 0] * 0.1 + X_m[:, 0] + 0.05 * rng.standard_normal(n)
    ds = _DS(y)
    levels = [0.1, 0.5, 0.9]
    moe = PhysGatedMorganQuantileMoE(
        n_components_gmm=2,
        quantile_levels=levels,
        n_ensemble_members=2,
        gmm_kwargs={"covariance_type": "diag", "n_init": 2},
        hgb_kwargs={"max_iter": 40, "max_depth": 3},
        random_state=0,
    )
    moe.fit(ds, X_gate=gate, X_morgan=X_m)
    q = moe.predict_quantiles(ds, X_gate=gate, X_morgan=X_m)
    assert q.shape == (n, 1, 3)
    Z = moe._gate_pre.transform(gate)
    pi = moe._gmm.predict_proba(Z)
    assert np.allclose(pi.sum(axis=1), 1.0, atol=1e-5)
    med = moe.predict(ds, X_gate=gate, X_morgan=X_m)
    assert med.shape == (n, 1)
    pb = moe.evaluate_pinball_loss(ds, X_gate=gate, X_morgan=X_m)
    assert np.isfinite(pb)


def test_phys_gated_moe_predict_requires_median_quantile():
    rng = np.random.default_rng(3)
    n = 50
    gate = rng.standard_normal((n, 3))
    X_m = rng.standard_normal((n, 8))
    y = rng.standard_normal(n)
    ds = _DS(y)
    moe = PhysGatedMorganQuantileMoE(
        n_components_gmm=2,
        quantile_levels=[0.1, 0.9],
        n_ensemble_members=1,
        gmm_kwargs={"covariance_type": "diag"},
        hgb_kwargs={"max_iter": 20},
        random_state=0,
    )
    moe.fit(ds, X_gate=gate, X_morgan=X_m)
    with pytest.raises(ValueError, match="0.5"):
        moe.predict(ds, X_gate=gate, X_morgan=X_m)
