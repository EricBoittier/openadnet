"""PyG encoders for molecular graphs (2D), inspired by DeepChem torch_models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    GraphConv,
    NNConv,
    global_mean_pool,
)


def _forward_pool_head(
    x: torch.Tensor,
    batch: torch.Tensor,
    head: nn.Linear,
) -> torch.Tensor:
    x = global_mean_pool(x, batch)
    return head(x)


class GINMolNet(nn.Module):
    """GIN (Xu et al.) with mean pooling."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_tasks: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_ch, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.head = nn.Linear(hidden_dim, n_tasks)

    def forward(self, batch) -> torch.Tensor:
        x, edge_index, b = batch.x, batch.edge_index, batch.batch
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return _forward_pool_head(x, b, self.head)


class GCNMolNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_tasks: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_ch, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.head = nn.Linear(hidden_dim, n_tasks)

    def forward(self, batch) -> torch.Tensor:
        x, edge_index, b = batch.x, batch.edge_index, batch.batch
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return _forward_pool_head(x, b, self.head)


class GATMolNet(nn.Module):
    """Multi-head GAT stack; last layer uses a single head and ``concat=False``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_tasks: int,
        num_layers: int,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        cin = in_dim
        for i in range(num_layers):
            last = i == num_layers - 1
            if last:
                self.convs.append(GATConv(cin, hidden_dim, heads=1, concat=False))
                cout = hidden_dim
            else:
                self.convs.append(GATConv(cin, hidden_dim, heads=heads, concat=True))
                cout = hidden_dim * heads
            self.batch_norms.append(nn.BatchNorm1d(cout))
            cin = cout
        self.head = nn.Linear(cin, n_tasks)

    def forward(self, batch) -> torch.Tensor:
        x, edge_index, b = batch.x, batch.edge_index, batch.batch
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return _forward_pool_head(x, b, self.head)


class GraphConvMolNet(nn.Module):
    """DeepChem-style GraphConv stack (PyG ``GraphConv``)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_tasks: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            self.convs.append(GraphConv(in_ch, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.head = nn.Linear(hidden_dim, n_tasks)

    def forward(self, batch) -> torch.Tensor:
        x, edge_index, b = batch.x, batch.edge_index, batch.batch
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return _forward_pool_head(x, b, self.head)


class MPNNMolNet(nn.Module):
    """NNConv-style MPNN (Gilmer et al.) using edge attributes."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        n_tasks: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.lin_skip = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            out_ch = hidden_dim
            nn_edge = nn.Sequential(
                nn.Linear(edge_dim, in_ch * out_ch),
                nn.ReLU(),
            )
            self.convs.append(NNConv(in_ch, out_ch, nn_edge, aggr="mean"))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.head = nn.Linear(hidden_dim, n_tasks)

    def forward(self, batch) -> torch.Tensor:
        x, edge_index, edge_attr, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        if edge_index.numel() == 0:
            x = self.lin_skip(x)
            x = global_mean_pool(x, b)
            return self.head(x)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
        return _forward_pool_head(x, b, self.head)


class AttentiveFPStyleMolNet(nn.Module):
    """GAT stack + mean pool (simplified AttentiveFP-style readout)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_tasks: int,
        num_layers: int,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        cin = in_dim
        for i in range(num_layers):
            last = i == num_layers - 1
            self.convs.append(
                GATConv(cin, hidden_dim, heads=heads, concat=not last)
            )
            cout = hidden_dim if last else hidden_dim * heads
            self.batch_norms.append(nn.BatchNorm1d(cout))
            cin = cout
        self.head = nn.Linear(cin, n_tasks)

    def forward(self, batch) -> torch.Tensor:
        x, edge_index, b = batch.x, batch.edge_index, batch.batch
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return _forward_pool_head(x, b, self.head)
