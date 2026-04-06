"""Tests for :mod:`delta_learning`."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.linear_model import LinearRegression


class TestDeltaHelpers(unittest.TestCase):
    def test_compute_combine(self) -> None:
        from delta_learning import combine_predictions, compute_delta

        y = np.array([3.0, 5.0])
        b = np.array([1.0, 2.0])
        d = compute_delta(y, b)
        self.assertTrue(np.allclose(d, [2.0, 3.0]))
        self.assertTrue(np.allclose(combine_predictions(b, d), y))

    def test_shape_mismatch(self) -> None:
        from delta_learning import compute_delta

        with self.assertRaises(ValueError):
            compute_delta(np.array([1.0]), np.array([1.0, 2.0]))


class TestDeltaLearningRegressor(unittest.TestCase):
    def test_perfect_baseline_zero_residual(self) -> None:
        from delta_learning import DeltaLearningRegressor

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 3))
        y = X[:, 0] + 0.1 * rng.standard_normal(40)
        model = DeltaLearningRegressor(estimator=LinearRegression())
        model.fit(X, y, y_baseline=y.copy())
        pred = model.predict(X, y_baseline=y)
        self.assertLess(np.mean((pred - y) ** 2), 1e-10)

    def test_constant_correction(self) -> None:
        from delta_learning import DeltaLearningRegressor

        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 2))
        y = 2.5 * np.ones(30)
        baseline = np.zeros(30)
        model = DeltaLearningRegressor(estimator=LinearRegression())
        model.fit(X, y, y_baseline=baseline)
        pred = model.predict(X, y_baseline=baseline)
        self.assertTrue(np.allclose(pred, y, rtol=1e-5))

    def test_predict_shape_row_mismatch(self) -> None:
        from delta_learning import DeltaLearningRegressor

        X = np.ones((5, 2))
        y = np.zeros(5)
        m = DeltaLearningRegressor(estimator=LinearRegression())
        m.fit(X, y, y_baseline=y)
        with self.assertRaises(ValueError):
            m.predict(X[:3], y_baseline=np.zeros(2))


if __name__ == "__main__":
    unittest.main()
