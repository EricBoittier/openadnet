"""Tests for :mod:`models.ensemble` (numpy only; no DL deps)."""

from __future__ import annotations

import unittest

import numpy as np


class _FakeRegressor:
    def __init__(self, n_tasks: int, fill: float) -> None:
        self.n_tasks = n_tasks
        self._fill = fill

    def fit(self, dataset, **kwargs):  # noqa: ANN001
        return []

    def predict(self, dataset, **kwargs):  # noqa: ANN001
        n = len(dataset)
        return np.full((n, self.n_tasks), self._fill, dtype=np.float64)


class _NoFit:
    n_tasks = 1

    def predict(self, dataset, **kwargs):  # noqa: ANN001
        return np.zeros((len(dataset), 1))


class TestEnsembleRegressor(unittest.TestCase):
    def test_weighted_mean(self) -> None:
        from models.ensemble import EnsembleRegressor

        class _DS:
            y = np.array([[0.0], [4.0]])

            def __len__(self) -> int:
                return 2

        ds = _DS()
        e = EnsembleRegressor(
            [_FakeRegressor(1, 0.0), _FakeRegressor(1, 4.0)],
            weights=[0.25, 0.75],
        )
        p = e.predict(ds)
        self.assertTrue(np.allclose(p, [[3.0], [3.0]]))
        self.assertAlmostEqual(e.evaluate_loss(ds), np.mean((p - ds.y) ** 2))

    def test_uniform_weights(self) -> None:
        from models.ensemble import EnsembleRegressor

        class _DS:
            def __len__(self) -> int:
                return 1

        ds = _DS()
        e = EnsembleRegressor([_FakeRegressor(2, 1.0), _FakeRegressor(2, 3.0)])
        p = e.predict(ds)
        self.assertTrue(np.allclose(p, [[2.0, 2.0]]))

    def test_n_tasks_mismatch(self) -> None:
        from models.ensemble import EnsembleRegressor

        with self.assertRaises(ValueError):
            EnsembleRegressor([_FakeRegressor(1, 0.0), _FakeRegressor(2, 0.0)])

    def test_fit_requires_callable(self) -> None:
        from models.ensemble import EnsembleRegressor

        class _DS:
            pass

        e = EnsembleRegressor([_NoFit()])
        with self.assertRaises(TypeError):
            e.fit(_DS())

    def test_fit_delegates(self) -> None:
        from models.ensemble import EnsembleRegressor

        class _FitModel:
            n_tasks = 1

            def __init__(self) -> None:
                self.called = False

            def fit(self, ds, **kwargs):  # noqa: ANN001
                self.called = True
                return [0.1]

            def predict(self, dataset, **kwargs):  # noqa: ANN001
                return np.zeros((len(dataset), 1))

        class _DS:
            pass

        m = _FitModel()
        e = EnsembleRegressor([m])
        h = e.fit(_DS())
        self.assertTrue(m.called)
        self.assertEqual(h, [[0.1]])


class TestMixedGnnTreeEnsemble(unittest.TestCase):
    def test_fingerprint_and_graph_member(self) -> None:
        from sklearn.linear_model import LinearRegression

        from models.ensemble import EnsembleRegressor, FingerprintEnsembleMember

        class _DS:
            def __init__(self, y: np.ndarray) -> None:
                self.y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

            def __len__(self) -> int:
                return len(self.y)

        ds = _DS(np.array([1.0, 3.0]))
        X_fp = np.ones((2, 1), dtype=np.float64)
        fp = FingerprintEnsembleMember("hgb", LinearRegression(), n_tasks=1)
        e = EnsembleRegressor([_FakeRegressor(1, 2.0), fp], weights=[0.5, 0.5])
        e.fit(ds, X_fp=X_fp)
        p = e.predict(ds, X_fp=X_fp)
        self.assertEqual(p.shape, (2, 1))
        self.assertTrue(np.allclose(p, 2.0))

    def test_mixed_fit_requires_x_fp(self) -> None:
        from sklearn.linear_model import LinearRegression

        from models.ensemble import EnsembleRegressor, FingerprintEnsembleMember

        class _DS:
            y = np.zeros((2, 1))

            def __len__(self) -> int:
                return 2

        e = EnsembleRegressor(
            [_FakeRegressor(1, 0.0), FingerprintEnsembleMember("hgb", LinearRegression())]
        )
        with self.assertRaises(ValueError):
            e.fit(_DS())

    def test_mixed_predict_requires_x_fp(self) -> None:
        from sklearn.linear_model import LinearRegression

        from models.ensemble import EnsembleRegressor, FingerprintEnsembleMember

        class _DS:
            y = np.array([[1.0], [2.0]])

            def __len__(self) -> int:
                return 2

        ds = _DS()
        X_fp = np.ones((2, 1))
        e = EnsembleRegressor(
            [_FakeRegressor(1, 1.0), FingerprintEnsembleMember("hgb", LinearRegression())]
        )
        e.fit(ds, X_fp=X_fp)
        with self.assertRaises(ValueError):
            e.predict(ds)


class _FakeQR:
    def __init__(self, n_tasks: int, fill: np.ndarray) -> None:
        self.n_tasks = n_tasks
        self._fill = np.asarray(fill, dtype=np.float64)

    def predict_quantiles(self, dataset, **kwargs):  # noqa: ANN001
        n = len(dataset)
        q = self._fill
        return np.broadcast_to(q, (n, self.n_tasks, q.shape[-1])).copy()


class TestEnsembleQuantileRegressor(unittest.TestCase):
    def test_weighted_quantiles(self) -> None:
        from models.ensemble import EnsembleQuantileRegressor

        class _DS:
            y = np.array([[1.0], [3.0]])

            def __len__(self) -> int:
                return 2

        ds = _DS()
        # per-member (n_tasks=1, n_q=3): low, median, high
        a = np.array([[[0.0, 1.0, 2.0]]])  # shape (1,1,3) broadcast to n=2
        b = np.array([[[4.0, 5.0, 6.0]]])
        e = EnsembleQuantileRegressor(
            [_FakeQR(1, a), _FakeQR(1, b)],
            quantile_levels=[0.1, 0.5, 0.9],
            weights=[0.25, 0.75],
        )
        p = e.predict_quantiles(ds)
        self.assertEqual(p.shape, (2, 1, 3))
        want = 0.25 * np.array([0.0, 1.0, 2.0]) + 0.75 * np.array([4.0, 5.0, 6.0])
        self.assertTrue(np.allclose(p[0, 0], want))
        self.assertTrue(np.allclose(e.predict(ds), p[:, :, 1]))

    def test_predict_requires_median(self) -> None:
        from models.ensemble import EnsembleQuantileRegressor

        class _DS:
            def __len__(self) -> int:
                return 1

        e = EnsembleQuantileRegressor(
            [_FakeQR(1, np.ones((1, 1, 2)))],
            quantile_levels=[0.1, 0.9],
        )
        with self.assertRaises(ValueError):
            e.predict(_DS())

    def test_pinball_loss_helper(self) -> None:
        from models.ensemble import pinball_loss

        y = np.array([0.0, 2.0])
        p = np.array([1.0, 1.0])
        # At q=0.5, pinball equals 0.5 * MAE
        self.assertAlmostEqual(pinball_loss(y, p, 0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
