"""Smoke tests for ConformalizedQuantileRegressor."""

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from uncertainty import ConformalizedQuantileRegressor, cross_validate_conformal_quantile


def test_cqr_fit_predict_interval_shape():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 3))
    y = X[:, 0] + 0.1 * rng.standard_normal(n)

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_fit, y_fit, test_size=0.2, random_state=1
    )

    base = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.5,
                    max_iter=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=0,
                ),
            ),
        ]
    )

    cqr = ConformalizedQuantileRegressor(estimator=base, alpha=0.1)
    cqr.fit(X_train, y_train, X_cal, y_cal)

    y_pred = cqr.predict(X_test)
    interval = cqr.predict_interval(X_test)
    y_full, interval2 = cqr.predict_full(X_test)

    assert y_pred.shape == (X_test.shape[0],)
    assert interval.shape == (X_test.shape[0], 2)
    assert np.allclose(interval, interval2)
    assert np.all(interval[:, 0] <= interval[:, 1])
    assert cqr.conformal_offset_ >= 0.0

    cov = np.mean((y_test >= interval[:, 0]) & (y_test <= interval[:, 1]))
    assert 0.5 <= cov <= 1.0


def test_cross_validate_conformal_quantile_kfold_length():
    rng = np.random.default_rng(1)
    n = 80
    X = rng.standard_normal((n, 2))
    y = X[:, 0] + 0.05 * rng.standard_normal(n)
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.5,
                    max_iter=30,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=0,
                ),
            ),
        ]
    )
    cqr = ConformalizedQuantileRegressor(estimator=pipe, alpha=0.1)
    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    out = cross_validate_conformal_quantile(
        cqr,
        X,
        y,
        cv,
        calibration_fraction=0.2,
        random_state=0,
        n_jobs=1,
    )
    assert out["test_rmse"].shape == (4,)
    assert out["test_mae"].shape == (4,)
    assert out["test_r2"].shape == (4,)
    assert np.all(np.isfinite(out["test_rmse"]))


def test_cqr_requires_nonempty_calibration():
    base = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.5, max_iter=5, random_state=0
    )
    cqr = ConformalizedQuantileRegressor(estimator=base, alpha=0.1)
    X = np.zeros((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="non-empty"):
        cqr.fit(X, y, X_cal=np.empty((0, 2)), y_cal=np.array([]))
