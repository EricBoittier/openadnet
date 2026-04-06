"""Tests for :mod:`models.ensemble` (numpy only; no DL deps)."""

from __future__ import annotations

import unittest

import numpy as np


class _FakeRegressor:
    def __init__(self, n_tasks: int, fill: float) -> None:
        self.n_tasks = n_tasks
        self._fill = fill

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


if __name__ == "__main__":
    unittest.main()
