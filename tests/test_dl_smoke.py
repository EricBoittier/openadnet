"""Smoke tests for ``models`` (requires ``pip install openadnet[dl]``)."""

from __future__ import annotations

import unittest


def _have_dl() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_have_dl(), "requires openadnet[dl]")
class TestDlDataTransformer(unittest.TestCase):
    def test_dataset_collate(self) -> None:
        import torch
        from transformers import AutoTokenizer

        from models.data import SmilesRegressionDataset, smiles_regression_collate_fn

        ds = SmilesRegressionDataset(["CCO", "CC"], [[1.0], [2.0]])
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")
        collate = smiles_regression_collate_fn(tok, max_length=32, return_labels=True)
        batch = collate([ds[0], ds[1]])
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        self.assertEqual(batch["labels"].shape, (2, 1))
        self.assertIsInstance(batch["labels"], torch.Tensor)

    def test_target_scaler(self) -> None:
        import numpy as np

        from models.data import TargetScaler

        y = np.array([[1.0], [3.0]])
        s = TargetScaler()
        z = s.fit_transform(y)
        self.assertAlmostEqual(float(z.mean()), 0.0, places=5)
        back = s.inverse_transform(z)
        self.assertTrue(np.allclose(back, y))


@unittest.skipUnless(_have_dl(), "requires openadnet[dl]")
class TestDlDataGraph(unittest.TestCase):
    def test_smiles_to_pyg(self) -> None:
        from models.data.graph import ATOM_FEATURE_DIM, smiles_to_pyg_data

        from models.data.graph import EDGE_FEATURE_DIM

        d = smiles_to_pyg_data("CCO", y=[0.5, 1.5])
        self.assertEqual(d.x.shape[1], ATOM_FEATURE_DIM)
        self.assertEqual(d.edge_attr.shape[1], EDGE_FEATURE_DIM)
        self.assertEqual(d.y.shape, (1, 2))

    def test_graph_dataset_y_property(self) -> None:
        import numpy as np

        from models.data.graph import GraphRegressionDataset, train_val_split_graph

        ds = GraphRegressionDataset(["CCO", "CCN"], np.array([[1.0], [2.0]]))
        self.assertTrue(np.allclose(ds.y, [[1.0], [2.0]]))
        tr, va = train_val_split_graph(ds, val_fraction=0.5, random_state=0)
        self.assertEqual(tr.y.shape[0], len(tr))
        self.assertEqual(va.y.shape[0], len(va))


@unittest.skipUnless(_have_dl(), "requires openadnet[dl]")
class TestGNNRegressor(unittest.TestCase):
    def test_fit_predict(self) -> None:
        import pandas as pd

        from models import GNNRegressor
        from models.data import graph_regression_from_dataframe

        df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"], "t": [1.0, 2.0, 3.0]})
        ds = graph_regression_from_dataframe(df, "smiles", ["t"])
        model = GNNRegressor(n_tasks=1, hidden_dim=32, num_layers=2)
        model.fit(ds, epochs=1, batch_size=2, show_progress=False)
        pred = model.predict(ds, show_progress=False)
        self.assertEqual(pred.shape, (3, 1))


@unittest.skipUnless(_have_dl(), "requires openadnet[dl]")
class TestHuggingFaceRegressor(unittest.TestCase):
    def test_fit_predict(self) -> None:
        import pandas as pd

        from models import HuggingFaceRegressor
        from models.data import smiles_regression_from_dataframe

        try:
            model = HuggingFaceRegressor(
                "hf-internal-testing/tiny-random-bert",
                n_tasks=1,
                max_length=32,
            )
        except OSError as e:
            self.skipTest(f"Hugging Face hub unavailable: {e}")
        df = pd.DataFrame({"smiles": ["CCO", "CCN"], "t": [1.0, 2.0]})
        ds = smiles_regression_from_dataframe(df, "smiles", ["t"])
        model.fit(ds, epochs=1, batch_size=2, show_progress=False)
        pred = model.predict(ds, show_progress=False)
        self.assertEqual(pred.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
