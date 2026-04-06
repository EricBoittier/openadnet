"""Factory for PyG molecular encoders."""

from __future__ import annotations

import torch.nn as nn

from models.nn.pyg_architectures import (
    AttentiveFPStyleMolNet,
    GATMolNet,
    GCNMolNet,
    GINMolNet,
    GraphConvMolNet,
    MPNNMolNet,
)

ARCHITECTURES: tuple[str, ...] = (
    "gin",
    "gcn",
    "gat",
    "graphconv",
    "mpnn",
    "attentivefp",
)


def create_pyg_module(
    architecture: str,
    *,
    in_dim: int,
    edge_dim: int,
    hidden_dim: int,
    n_tasks: int,
    num_layers: int,
    gat_heads: int = 4,
) -> nn.Module:
    """Build a PyG encoder; ``forward(batch)`` expects a batched PyG ``Data`` object."""
    key = architecture.lower()
    if key == "gin":
        return GINMolNet(in_dim, hidden_dim, n_tasks, num_layers)
    if key == "gcn":
        return GCNMolNet(in_dim, hidden_dim, n_tasks, num_layers)
    if key == "gat":
        return GATMolNet(in_dim, hidden_dim, n_tasks, num_layers, heads=gat_heads)
    if key in ("graphconv", "graph_conv"):
        return GraphConvMolNet(in_dim, hidden_dim, n_tasks, num_layers)
    if key == "mpnn":
        return MPNNMolNet(in_dim, edge_dim, hidden_dim, n_tasks, num_layers)
    if key in ("attentivefp", "attentive_fp"):
        return AttentiveFPStyleMolNet(
            in_dim, hidden_dim, n_tasks, num_layers, heads=gat_heads
        )
    raise ValueError(
        f"unknown architecture {architecture!r}; expected one of {ARCHITECTURES}"
    )


def create_pyg_regressor(**kwargs):
    """Alias for :class:`~models.nn.pyg_regressor.PyGMoleculeRegressor` with the same kwargs."""
    from models.nn.pyg_regressor import PyGMoleculeRegressor

    return PyGMoleculeRegressor(**kwargs)
