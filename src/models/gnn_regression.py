"""PyTorch Geometric GNN encoders with a regression head (default: GAT)."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from models.nn.pyg_regressor import PyGMoleculeRegressor


class GNNRegressor(PyGMoleculeRegressor):
    """Train and predict on :class:`~models.data.graph.GraphRegressionDataset` (PyG).

    Convenience subclass of :class:`~models.nn.pyg_regressor.PyGMoleculeRegressor` with
    default **GAT** (``architecture="gat"``). Pass ``architecture=...`` for any encoder
    supported by :func:`~models.nn.registry.create_pyg_module` (e.g. ``"gin"``, ``"gcn"``,
    ``"gat"``, ``"graphconv"``, ``"mpnn"``, ``"attentivefp"``).

    ``gat`` and ``attentivefp`` use ``gat_heads`` (default ``4``).
    """

    def __init__(
        self,
        n_tasks: int,
        *,
        architecture: str = "gat",
        in_dim: Optional[int] = None,
        descriptor_name: Optional[Union[str, Sequence[str]]] = None,
        edge_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 3,
        gat_heads: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            n_tasks,
            architecture=architecture,
            in_dim=in_dim,
            descriptor_name=descriptor_name,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gat_heads=gat_heads,
            device=device,
        )
