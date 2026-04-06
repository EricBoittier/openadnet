"""Smoke tests for PyG molecular encoders (requires ``openadnet[dl]``)."""

from __future__ import annotations

import unittest


def _have_dl() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_have_dl(), "requires openadnet[dl]")
class TestPyGArchitectures(unittest.TestCase):
    def _batch_cco(self):
        import torch
        from torch_geometric.data import Batch

        from models.data.graph import EDGE_FEATURE_DIM, smiles_to_pyg_data

        d = smiles_to_pyg_data("CCO", y=[1.0])
        self.assertEqual(d.edge_attr.shape[1], EDGE_FEATURE_DIM)
        return Batch.from_data_list([d])

    def test_forward_each(self) -> None:
        import torch

        from models.data.graph import atom_feature_dim_default
        from models.nn.registry import ARCHITECTURES, create_pyg_module

        batch = self._batch_cco()
        in_dim = atom_feature_dim_default()
        edge_dim = batch.edge_attr.shape[1]
        device = torch.device("cpu")
        for name in ARCHITECTURES:
            with self.subTest(arch=name):
                m = create_pyg_module(
                    name,
                    in_dim=in_dim,
                    edge_dim=edge_dim,
                    hidden_dim=32,
                    n_tasks=1,
                    num_layers=2,
                ).to(device)
                batch_d = batch.to(device)
                out = m(batch_d)
                self.assertEqual(out.shape, (1, 1))

    def test_mpnn_edgeless_graph(self) -> None:
        """MPNN skips convs when there are no edges (e.g. single atom)."""
        import torch
        from torch_geometric.data import Batch, Data

        from models.data.graph import EDGE_FEATURE_DIM, atom_feature_dim_default
        from models.nn.registry import create_pyg_module

        x = torch.randn(1, atom_feature_dim_default())
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([[0.0]]))
        batch = Batch.from_data_list([d])
        m = create_pyg_module(
            "mpnn",
            in_dim=x.shape[1],
            edge_dim=EDGE_FEATURE_DIM,
            hidden_dim=16,
            n_tasks=1,
            num_layers=2,
        )
        out = m(batch)
        self.assertEqual(out.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
