"""RDKit molecular graphs as :class:`torch_geometric.data.Data` for regression."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import HybridizationType
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# One-hot atom types (common in medchem + H)
_ATOM_ORDER = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other"]
_ATOM_DIM = len(_ATOM_ORDER)
_DEGREE_DIM = 6
_FORMAL_CHARGE_DIM = 11  # -5 .. +5
_HYB_DIM = len(tuple(HybridizationType.values))  # includes UNSPECIFIED

# 11 + 6 + 11 + 1 (aromatic) + hyb
ATOM_FEATURE_DIM: int = _ATOM_DIM + _DEGREE_DIM + _FORMAL_CHARGE_DIM + 1 + _HYB_DIM

# Bond: type one-hot (6) + conjugated + in_ring
_BOND_TYPE_DIM = 6
EDGE_FEATURE_DIM: int = _BOND_TYPE_DIM + 2


def atom_feature_dim_default() -> int:
    return ATOM_FEATURE_DIM


def edge_feature_dim_default() -> int:
    return EDGE_FEATURE_DIM


def _one_hot(idx: int, dim: int) -> List[float]:
    v = [0.0] * dim
    if 0 <= idx < dim:
        v[idx] = 1.0
    return v


def _atom_type_index(symbol: str) -> int:
    if symbol in _ATOM_ORDER[:-1]:
        return _ATOM_ORDER.index(symbol)
    return _ATOM_ORDER.index("other")


def _hybrid_index(h: Chem.rdchem.HybridizationType) -> int:
    vals = list(HybridizationType.values)
    try:
        return vals.index(h)
    except ValueError:
        return vals.index(HybridizationType.UNSPECIFIED)  # type: ignore[arg-type]


def _bond_type_index(bond: Chem.Bond) -> int:
    t = bond.GetBondType()
    order = (
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        Chem.rdchem.BondType.DATIVE,
    )
    try:
        return order.index(t)
    except ValueError:
        return _BOND_TYPE_DIM - 1


def _bond_features(bond: Chem.Bond) -> List[float]:
    feat = _one_hot(_bond_type_index(bond), _BOND_TYPE_DIM)
    feat.append(1.0 if bond.GetIsConjugated() else 0.0)
    feat.append(1.0 if bond.IsInRing() else 0.0)
    return feat


def _atom_features(atom: Chem.Atom) -> List[float]:
    sym = atom.GetSymbol()
    t = _atom_type_index(sym)
    feat = _one_hot(t, _ATOM_DIM)
    deg = min(atom.GetDegree(), _DEGREE_DIM - 1)
    feat += _one_hot(deg, _DEGREE_DIM)
    fc = atom.GetFormalCharge()
    fc_idx = min(max(fc + 5, 0), _FORMAL_CHARGE_DIM - 1)
    feat += _one_hot(fc_idx, _FORMAL_CHARGE_DIM)
    feat.append(1.0 if atom.GetIsAromatic() else 0.0)
    feat += _one_hot(_hybrid_index(atom.GetHybridization()), _HYB_DIM)
    return feat


def mol_to_pyg_data(mol: Chem.Mol, y: Optional[np.ndarray] = None) -> Data:
    """Convert a sanitized RDKit molecule to a single PyG ``Data`` object."""
    if mol.GetNumAtoms() == 0:
        raise ValueError("empty molecule")
    xs: List[List[float]] = []
    for atom in mol.GetAtoms():
        xs.append(_atom_features(atom))
    x = torch.tensor(xs, dtype=torch.float32)

    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_feats: List[List[float]] = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        edge_src += [a, b]
        edge_dst += [b, a]
        edge_feats += [bf, bf]
    if not edge_src:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        # Shape (1, n_tasks) so PyG ``Batch`` stacks to (batch_size, n_tasks).
        yy = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(1, -1)
        data.y = yy
    return data


def smiles_to_pyg_data(smiles: str, y: Optional[np.ndarray] = None) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    try:
        Chem.SanitizeMol(mol)
    except Chem.MolSanitizeException as e:
        raise ValueError(f"sanitize failed for SMILES: {smiles!r}") from e
    return mol_to_pyg_data(mol, y=y)


class GraphRegressionDataset(torch.utils.data.Dataset):
    """One graph per SMILES with optional multi-task regression targets."""

    def __init__(
        self,
        smiles: Sequence[str],
        y: np.ndarray,
        *,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self._smiles = list(smiles)
        self._y = np.asarray(y, dtype=np.float64)
        if self._y.ndim == 1:
            self._y = self._y.reshape(-1, 1)
        if len(self._smiles) != len(self._y):
            raise ValueError("smiles and y must have the same length")
        if indices is None:
            self._indices = list(range(len(self._smiles)))
        else:
            self._indices = list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Data:
        i = self._indices[idx]
        s = self._smiles[i]
        row_y = self._y[i]
        return smiles_to_pyg_data(s, y=row_y)

    @property
    def n_tasks(self) -> int:
        return int(self._y.shape[1])

    @property
    def smiles(self) -> List[str]:
        return [self._smiles[i] for i in self._indices]

    @property
    def row_indices(self) -> List[int]:
        return list(self._indices)

    @property
    def y(self) -> np.ndarray:
        """Targets for samples in this dataset (same order as ``__getitem__`` / iteration)."""
        idx = np.asarray(self._indices, dtype=np.intp)
        return np.asarray(self._y[idx], dtype=np.float64)


def graph_regression_from_dataframe(
    df: pd.DataFrame,
    smiles_col: str,
    target_cols: Sequence[str],
    drop_na_targets: bool = True,
) -> GraphRegressionDataset:
    work = df[[smiles_col, *target_cols]].copy()
    if drop_na_targets:
        work = work.dropna(subset=list(target_cols))
    y = work[list(target_cols)].to_numpy(dtype=np.float64)
    smiles = work[smiles_col].astype(str).tolist()
    keep_smiles: List[str] = []
    keep_y: List[np.ndarray] = []
    for i, s in enumerate(smiles):
        if Chem.MolFromSmiles(s) is None:
            continue
        try:
            smiles_to_pyg_data(s)
        except ValueError:
            continue
        keep_smiles.append(s)
        keep_y.append(y[i])
    if not keep_smiles:
        raise ValueError("no valid SMILES in dataframe")
    y_arr = np.stack(keep_y, axis=0)
    return GraphRegressionDataset(keep_smiles, y_arr)


def train_val_split_graph(
    dataset: GraphRegressionDataset,
    val_fraction: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[GraphRegressionDataset, GraphRegressionDataset]:
    """Random train/validation split over graphs in ``dataset``."""
    n = len(dataset)
    if n < 2:
        raise ValueError("need at least 2 samples to split")
    idx = np.arange(n)
    train_pos, val_pos = train_test_split(
        idx,
        test_size=val_fraction,
        random_state=random_state,
    )
    train_rows = [dataset.row_indices[int(i)] for i in train_pos]
    val_rows = [dataset.row_indices[int(i)] for i in val_pos]
    smiles_all = dataset._smiles  # type: ignore[attr-defined]
    y_all = dataset._y  # type: ignore[attr-defined]
    train_ds = GraphRegressionDataset(smiles_all, y_all, indices=train_rows)
    val_ds = GraphRegressionDataset(smiles_all, y_all, indices=val_rows)
    return train_ds, val_ds
