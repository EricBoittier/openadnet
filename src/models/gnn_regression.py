"""PyTorch Geometric GNN encoders with a regression head (default: GIN)."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from models.nn.pyg_regressor import PyGMoleculeRegressor


class GNNRegressor(PyGMoleculeRegressor):
    """Train and predict on :class:`~models.data.graph.GraphRegressionDataset` (PyG).

    This class is the backward-compatible entry point: it defaults to a **GIN** encoder.
    For other architectures (GCN, GAT, MPNN, etc.), use
    :class:`~models.nn.pyg_regressor.PyGMoleculeRegressor` with ``architecture=...``.
    """

    def __init__(
        self,
        n_tasks: int,
        *,
        in_dim: Optional[int] = None,
        descriptor_name: Optional[Union[str, Sequence[str]]] = None,
        hidden_dim: int = 64,
        num_layers: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            n_tasks,
            architecture="gin",
            in_dim=in_dim,
            descriptor_name=descriptor_name,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device,
        )
