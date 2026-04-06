"""Train/predict wrapper for PyG molecular encoders (``forward(batch)``)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.data.graph import (
    GraphRegressionDataset,
    atom_feature_dim_default,
    atom_feature_dim_with_descriptor,
    coerce_graph_descriptor_names,
    edge_feature_dim_default,
)
from models.nn.registry import create_pyg_module


def _descriptor_tuple_key(
    dn: Optional[Union[str, Tuple[str, ...]]],
) -> Optional[Tuple[str, ...]]:
    if dn is None:
        return None
    if isinstance(dn, str):
        return (dn,)
    return tuple(dn)


class PyGMoleculeRegressor:
    """Train and predict on :class:`~models.data.graph.GraphRegressionDataset` using a PyG encoder."""

    def __init__(
        self,
        n_tasks: int,
        *,
        architecture: str = "gin",
        in_dim: Optional[int] = None,
        descriptor_name: Optional[Union[str, Sequence[str]]] = None,
        edge_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 3,
        gat_heads: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        if n_tasks < 1:
            raise ValueError("n_tasks must be >= 1")
        self.n_tasks = n_tasks
        self.architecture = architecture.lower()
        self._descriptor_names = coerce_graph_descriptor_names(descriptor_name)
        if in_dim is not None:
            self.in_dim = in_dim
        elif self._descriptor_names is not None:
            self.in_dim = atom_feature_dim_with_descriptor(self._descriptor_names)
        else:
            self.in_dim = atom_feature_dim_default()
        self.edge_dim = edge_dim if edge_dim is not None else edge_feature_dim_default()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gat_heads = gat_heads
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = create_pyg_module(
            self.architecture,
            in_dim=self.in_dim,
            edge_dim=self.edge_dim,
            hidden_dim=hidden_dim,
            n_tasks=n_tasks,
            num_layers=num_layers,
            gat_heads=gat_heads,
        ).to(self.device)

    @property
    def descriptor_name(self) -> Optional[Union[str, Tuple[str, ...]]]:
        if self._descriptor_names is None:
            return None
        if len(self._descriptor_names) == 1:
            return self._descriptor_names[0]
        return self._descriptor_names

    def _assert_dataset_descriptor(self, dataset: GraphRegressionDataset) -> None:
        ds_dn = getattr(dataset, "descriptor_name", None)
        if _descriptor_tuple_key(ds_dn) != self._descriptor_names:
            raise ValueError(
                f"dataset descriptor_name={ds_dn!r} does not match model "
                f"{self.descriptor_name!r}"
            )

    def fit(
        self,
        train_dataset: GraphRegressionDataset,
        *,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        val_dataset: Optional[GraphRegressionDataset] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        lr_reduce_factor: float = 0.5,
        max_lr_reductions: int = 5,
        min_lr: float = 1e-8,
        show_progress: bool = True,
    ) -> List[float]:
        """Train the encoder; optionally monitor ``val_dataset`` each epoch.

        When ``val_dataset`` is set, validation MSE is computed after each epoch
        and the best weights (lowest validation loss) are restored at the end.

        If validation does not improve for ``early_stopping_patience`` epochs
        (default ``10`` when ``early_stopping_patience`` is ``None``), the
        learning rate is multiplied by ``lr_reduce_factor`` (clamped to at least
        ``min_lr``), the no-improvement counter resets, and training continues.
        After ``max_lr_reductions`` such LR decreases, training stops. If
        ``early_stopping_patience`` is ``0``, no LR reductions are applied and
        training runs for the full ``epochs`` (best validation checkpoint is
        still restored when ``val_dataset`` is provided).
        """
        if train_dataset.n_tasks != self.n_tasks:
            raise ValueError(
                f"dataset has n_tasks={train_dataset.n_tasks}, model expects {self.n_tasks}"
            )
        self._assert_dataset_descriptor(train_dataset)
        if val_dataset is not None:
            self._assert_dataset_descriptor(val_dataset)
        patience_eff = early_stopping_patience
        if val_dataset is not None and patience_eff is None:
            patience_eff = 10
        elif val_dataset is None:
            patience_eff = 0
        if max_lr_reductions < 1:
            raise ValueError("max_lr_reductions must be >= 1")
        if not (0.0 < lr_reduce_factor < 1.0):
            raise ValueError("lr_reduce_factor must be in (0, 1)")

        self.model.train()
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        losses: List[float] = []
        best_val = float("inf")
        best_state: Optional[dict[str, torch.Tensor]] = None
        epochs_no_improve = 0
        lr_reduce_count = 0
        epoch_bar = tqdm(range(epochs), disable=not show_progress, desc="epoch")
        for _ in epoch_bar:
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                pred = self.model(batch)
                target = batch.y
                loss = F.mse_loss(pred, target)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            avg = epoch_loss / max(n_batches, 1)
            losses.append(avg)
            cur_lr = float(opt.param_groups[0]["lr"])
            postfix: dict[str, float] = {"loss": avg, "lr": cur_lr}
            if val_dataset is not None:
                vloss = self.evaluate_loss(val_dataset, batch_size=batch_size)
                postfix["val_loss"] = vloss
                if vloss < best_val - early_stopping_min_delta:
                    best_val = vloss
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if patience_eff > 0 and epochs_no_improve >= patience_eff:
                    lr_reduce_count += 1
                    for pg in opt.param_groups:
                        pg["lr"] = max(float(pg["lr"]) * lr_reduce_factor, min_lr)
                    epochs_no_improve = 0
                    if show_progress:
                        tqdm.write(
                            f"val plateau: reducing lr to {opt.param_groups[0]['lr']:.2e} "
                            f"({lr_reduce_count}/{max_lr_reductions})"
                        )
                    if lr_reduce_count >= max_lr_reductions:
                        if show_progress:
                            tqdm.write(
                                f"stopping: reached max_lr_reductions={max_lr_reductions}"
                            )
                        break
            epoch_bar.set_postfix(**postfix)

        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()}
            )
        return losses

    @torch.no_grad()
    def evaluate_loss(
        self,
        dataset: GraphRegressionDataset,
        *,
        batch_size: int = 32,
    ) -> float:
        self._assert_dataset_descriptor(dataset)
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(self.device)
            pred = self.model(batch)
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
        self._assert_dataset_descriptor(dataset)
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        chunks: List[np.ndarray] = []
        it = tqdm(loader, disable=not show_progress, desc="predict")
        for batch in it:
            batch = batch.to(self.device)
            pred = self.model(batch)
            chunks.append(pred.detach().cpu().numpy())
        self.model.train()
        return np.concatenate(chunks, axis=0)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        if self._descriptor_names is None:
            saved_dn = None
        elif len(self._descriptor_names) == 1:
            saved_dn = self._descriptor_names[0]
        else:
            saved_dn = list(self._descriptor_names)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_tasks": self.n_tasks,
                "architecture": self.architecture,
                "in_dim": self.in_dim,
                "edge_dim": self.edge_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "gat_heads": self.gat_heads,
                "descriptor_name": saved_dn,
            },
            path / "gnn_regression.pt",
        )

    def load_pretrained(self, save_directory: Union[str, Path]) -> None:
        path = Path(save_directory) / "gnn_regression.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.n_tasks = int(ckpt["n_tasks"])
        self.architecture = str(ckpt.get("architecture", "gin")).lower()
        self.in_dim = int(ckpt["in_dim"])
        self.edge_dim = int(ckpt.get("edge_dim", edge_feature_dim_default()))
        self.hidden_dim = int(ckpt["hidden_dim"])
        self.num_layers = int(ckpt["num_layers"])
        self.gat_heads = int(ckpt.get("gat_heads", 4))
        raw_dn = ckpt.get("descriptor_name")
        self._descriptor_names = coerce_graph_descriptor_names(raw_dn)
        self.model = create_pyg_module(
            self.architecture,
            in_dim=self.in_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            n_tasks=self.n_tasks,
            num_layers=self.num_layers,
            gat_heads=self.gat_heads,
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
