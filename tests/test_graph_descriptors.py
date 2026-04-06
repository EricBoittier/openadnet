"""Tests for molecule-level descriptors fused into graph node features."""

from __future__ import annotations

import unittest

import numpy as np


def _have_dl() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError:
        return False
    return True


class TestDescriptorDim(unittest.TestCase):
    def test_dims(self) -> None:
        from features_data import descriptor_dim

        self.assertEqual(descriptor_dim("morgan_r2_bits_512"), 512)
        self.assertGreater(descriptor_dim("rdkit_phys_props"), 0)


@unittest.skipUnless(_have_dl(), "requires openadnet[dl]")
class TestMolExtraNodeFeat(unittest.TestCase):
    def test_broadcast_concat_shape(self) -> None:
        import torch
        from rdkit import Chem

        from models.data.graph import (
            ATOM_FEATURE_DIM,
            atom_feature_dim_with_descriptor,
            mol_to_pyg_data,
        )

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
        d = 8
        extra = np.random.randn(d).astype(np.float64)
        data = mol_to_pyg_data(mol, extra_node_feat=extra)
        n = mol.GetNumAtoms()
        self.assertEqual(data.x.shape, (n, ATOM_FEATURE_DIM + d))
        self.assertEqual(
            atom_feature_dim_with_descriptor("morgan_r2_bits_512"),
            ATOM_FEATURE_DIM + 512,
        )

    def test_explicit_per_atom_rows(self) -> None:
        from rdkit import Chem

        from models.data.graph import ATOM_FEATURE_DIM, mol_to_pyg_data

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
        n = mol.GetNumAtoms()
        d = 3
        extra = np.random.randn(n, d).astype(np.float64)
        data = mol_to_pyg_data(mol, extra_node_feat=extra)
        self.assertEqual(data.x.shape, (n, ATOM_FEATURE_DIM + d))


@unittest.skipUnless(_have_dl(), "requires openadnet[dl]")
class TestGraphDatasetDescriptor(unittest.TestCase):
    def test_dataset_and_forward(self) -> None:
        import pandas as pd
        import torch
        from torch_geometric.data import Batch

        from models.data import graph_regression_from_dataframe
        from models.nn.pyg_regressor import PyGMoleculeRegressor

        df = pd.DataFrame(
            {
                "smiles": ["CCO", "CCN", "CCC"],
                "t": [1.0, 2.0, 3.0],
            }
        )
        desc = "morgan_r2_bits_512"
        ds = graph_regression_from_dataframe(
            df, "smiles", ["t"], descriptor_name=desc
        )
        self.assertEqual(ds.descriptor_name, desc)
        d0 = ds[0]
        self.assertGreater(d0.x.shape[1], 512)

        model = PyGMoleculeRegressor(
            n_tasks=1,
            architecture="gin",
            descriptor_name=desc,
            hidden_dim=16,
            num_layers=2,
        )
        batch = Batch.from_data_list([ds[0]])
        batch = batch.to(torch.device("cpu"))
        out = model.model(batch)
        self.assertEqual(out.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
