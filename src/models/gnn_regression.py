"""PyTorch Geometric GIN encoder with a regression head."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from tqdm import tqdm

from models.data.graph import GraphRegressionDataset, atom_feature_dim_default


class _GINRegressionNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_tasks: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.head(x)


class GNNRegressor:
    """Train and predict on :class:`~models.data.graph.GraphRegressionDataset` (PyG)."""

    def __init__(
        self,
        n_tasks: int,
        *,
        in_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        if n_tasks < 1:
            raise ValueError("n_tasks must be >= 1")
        self.n_tasks = n_tasks
        self.in_dim = in_dim if in_dim is not None else atom_feature_dim_default()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = _GINRegressionNet(
            self.in_dim,
            hidden_dim,
            n_tasks,
            num_layers,
        ).to(self.device)

    def fit(
        self,
        train_dataset: GraphRegressionDataset,
        *,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        val_dataset: Optional[GraphRegressionDataset] = None,
        show_progress: bool = True,
    ) -> List[float]:
        if train_dataset.n_tasks != self.n_tasks:
            raise ValueError(
                f"dataset has n_tasks={train_dataset.n_tasks}, model expects {self.n_tasks}"
            )
        self.model.train()
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        losses: List[float] = []
        epoch_bar = tqdm(range(epochs), disable=not show_progress, desc="epoch")
        for _ in epoch_bar:
            batch_bar = tqdm(loader, disable=not show_progress, leave=False, desc="train")
            epoch_loss = 0.0
            n_batches = 0
            for batch in batch_bar:
                batch = batch.to(self.device)
                opt.zero_grad()
                pred = self.model(batch.x, batch.edge_index, batch.batch)
                target = batch.y
                loss = F.mse_loss(pred, target)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            losses.append(avg)
            epoch_bar.set_postfix(loss=avg)

        if val_dataset is not None:
            vloss = self.evaluate_loss(val_dataset, batch_size=batch_size)
            if show_progress:
                tqdm.write(f"validation loss: {vloss:.6f}")
        return losses

    @torch.no_grad()
    def evaluate_loss(
        self,
        dataset: GraphRegressionDataset,
        *,
        batch_size: int = 32,
    ) -> float:
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch.x, batch.edge_index, batch.batch)
            target = batch.y
            total += float(F.mse_loss(pred, target).cpu())
            n_batches += 1
        self.model.train()
        return total / max(n_batches, 1)

    @torch.no_grad()
    def predict(
        self,
        dataset: GraphRegressionDataset,
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        chunks: List[np.ndarray] = []
        it = tqdm(loader, disable=not show_progress, desc="predict")
        for batch in it:
            batch = batch.to(self.device)
            pred = self.model(batch.x, batch.edge_index, batch.batch)
            chunks.append(pred.detach().cpu().numpy())
        self.model.train()
        return np.concatenate(chunks, axis=0)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_tasks": self.n_tasks,
                "in_dim": self.in_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
            },
            path / "gnn_regression.pt",
        )

    def load_pretrained(self, save_directory: Union[str, Path]) -> None:
        path = Path(save_directory) / "gnn_regression.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.n_tasks = int(ckpt["n_tasks"])
        self.in_dim = int(ckpt["in_dim"])
        self.hidden_dim = int(ckpt["hidden_dim"])
        self.num_layers = int(ckpt["num_layers"])
        self.model = _GINRegressionNet(
            self.in_dim,
            self.hidden_dim,
            self.n_tasks,
            self.num_layers,
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
