"""SMILES + regression targets for Hugging Face transformer training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase


class TargetScaler:
    """Per-target standardization for regression labels (``y``)."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, y: np.ndarray) -> "TargetScaler":
        self._scaler.fit(y)
        self._fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TargetScaler.transform called before fit")
        return self._scaler.transform(y)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TargetScaler.inverse_transform called before fit")
        return self._scaler.inverse_transform(y)

    @property
    def fitted(self) -> bool:
        return self._fitted


class SmilesRegressionDataset:
    """Indexable dataset of SMILES strings and regression targets.

    ``__getitem__`` returns ``(smiles: str, y: np.ndarray)`` with ``y`` shape
    ``(n_tasks,)``. Invalid SMILES rows should be dropped before construction
    (see :func:`smiles_regression_from_dataframe`).
    """

    def __init__(
        self,
        smiles: Sequence[str],
        y: np.ndarray,
    ) -> None:
        if len(smiles) != len(y):
            raise ValueError("smiles and y must have the same length")
        self._smiles: List[str] = list(smiles)
        self._y = np.asarray(y, dtype=np.float64)
        if self._y.ndim == 1:
            self._y = self._y.reshape(-1, 1)

    def __len__(self) -> int:
        return len(self._smiles)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray]:
        return self._smiles[idx], self._y[idx].copy()

    @property
    def n_tasks(self) -> int:
        return int(self._y.shape[1])

    @property
    def smiles(self) -> List[str]:
        return list(self._smiles)

    @property
    def y(self) -> np.ndarray:
        return self._y


def smiles_regression_from_dataframe(
    df: pd.DataFrame,
    smiles_col: str,
    target_cols: Sequence[str],
    drop_na_targets: bool = True,
) -> SmilesRegressionDataset:
    """Build a :class:`SmilesRegressionDataset` from a table.

    Rows with missing targets are dropped when ``drop_na_targets`` is True.
    """
    work = df[[smiles_col, *target_cols]].copy()
    if drop_na_targets:
        work = work.dropna(subset=list(target_cols))
    y = work[list(target_cols)].to_numpy(dtype=np.float64)
    smiles = work[smiles_col].astype(str).tolist()
    return SmilesRegressionDataset(smiles, y)


def train_val_split_smiles(
    dataset: SmilesRegressionDataset,
    val_fraction: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[SmilesRegressionDataset, SmilesRegressionDataset]:
    """Random train/validation split preserving indexing semantics."""
    n = len(dataset)
    if n < 2:
        raise ValueError("need at least 2 samples to split")
    indices = np.arange(n)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=random_state,
    )
    y = dataset.y
    smiles = dataset.smiles
    train_ds = SmilesRegressionDataset(
        [smiles[i] for i in train_idx],
        y[train_idx],
    )
    val_ds = SmilesRegressionDataset(
        [smiles[i] for i in val_idx],
        y[val_idx],
    )
    return train_ds, val_ds


def smiles_regression_collate_fn(
    tokenizer: "PreTrainedTokenizerBase",
    max_length: Optional[int] = None,
    *,
    return_labels: bool = True,
) -> Callable[
    [List[Tuple[str, np.ndarray]]],
    dict,
]:
    """Build a collate function for :class:`torch.utils.data.DataLoader`.

    Returns a dict with tokenizer outputs and optionally ``labels`` (float tensor).
    Keys match Hugging Face ``AutoModelForSequenceClassification`` inputs.
    """

    import torch

    def collate(
        batch: List[Tuple[str, np.ndarray]],
    ) -> dict:
        smiles_batch, y_batch = zip(*batch)
        text = list(smiles_batch)
        enc = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out: dict = {k: v for k, v in enc.items()}
        if return_labels:
            y_stacked = np.stack(y_batch, axis=0)
            out["labels"] = torch.tensor(y_stacked, dtype=torch.float32)
        return out

    return collate
