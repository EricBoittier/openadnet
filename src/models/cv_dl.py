"""K-fold cross-validation for SMILES regression models (HF and PyG GNN)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from baseline import BaselineCVConfig
from models.data import graph_regression_from_dataframe, smiles_regression_from_dataframe
from models.nn.pyg_regressor import PyGMoleculeRegressor
from models.hf_regression import HuggingFaceRegressor


def prepare_regression_frame(
    train_df: pd.DataFrame,
    smiles_col: str,
    target_cols: list[str],
) -> pd.DataFrame:
    cols = [smiles_col, *target_cols]
    missing = [c for c in cols if c not in train_df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    work = train_df[cols].copy()
    work = work.dropna(subset=list(target_cols))
    work = work.reset_index(drop=True)
    if len(work) < 2:
        raise ValueError("need at least 2 rows with valid targets for CV")
    return work


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, and uniform-average R² (multi-task supported)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def summarize_cv_folds(fold_df: pd.DataFrame) -> pd.Series:
    """Mean and std of ``rmse``, ``mae``, ``r2`` across folds."""
    out: dict[str, float] = {}
    for col in ("rmse", "mae", "r2"):
        if col in fold_df.columns:
            out[f"mean_{col}"] = float(fold_df[col].mean())
            out[f"std_{col}"] = float(fold_df[col].std(ddof=0))
    return pd.Series(out)


def run_hf_regressor_cv(
    train_df: pd.DataFrame,
    smiles_col: str,
    target_cols: list[str],
    model_name_or_path: str,
    *,
    tokenizer_name_or_path: Optional[str] = None,
    config: Optional[BaselineCVConfig] = None,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: Optional[int] = 256,
    show_progress: bool = True,
    fit_show_progress: bool = False,
    device: Optional[torch.device] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """K-fold CV for :class:`~models.hf_regression.HuggingFaceRegressor`.

    ``show_progress`` controls the **fold** tqdm bar only. Per-epoch training bars from
    ``HuggingFaceRegressor.fit`` are off by default; set ``fit_show_progress=True`` to
    enable them (can be noisy with many folds).

    Use ``device=torch.device("cpu")`` or ``OPENADNET_FORCE_CPU=1`` if CUDA raises
    device-side asserts in notebooks.

    Returns
    -------
    fold_results
        One row per fold: ``fold``, ``rmse``, ``mae``, ``r2``, ``n_train``, ``n_val``.
    summary
        Mean/std of metrics across folds.
    """
    cfg = config or BaselineCVConfig()
    work = prepare_regression_frame(train_df, smiles_col, target_cols)
    n_tasks = len(target_cols)
    kf = KFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=cfg.cv_random_state,
    )
    rows: list[dict[str, float | int]] = []
    fold_iter = enumerate(kf.split(np.arange(len(work))))
    if show_progress:
        fold_iter = tqdm(
            list(fold_iter),
            desc="hf_cv",
            unit="fold",
        )
    for fold_id, (train_idx, val_idx) in fold_iter:
        train_part = work.iloc[train_idx]
        val_part = work.iloc[val_idx]
        train_ds = smiles_regression_from_dataframe(
            train_part, smiles_col, target_cols
        )
        val_ds = smiles_regression_from_dataframe(val_part, smiles_col, target_cols)
        model = HuggingFaceRegressor(
            model_name_or_path,
            n_tasks=n_tasks,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_length=max_length,
            device=device,
        )
        model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            val_dataset=val_ds,
            show_progress=fit_show_progress,
        )
        y_pred = model.predict(val_ds, show_progress=False)
        y_true = val_ds.y
        m = regression_metrics(y_true, y_pred)
        rows.append(
            {
                "fold": fold_id,
                **m,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            }
        )
    fold_df = pd.DataFrame(rows)
    return fold_df, summarize_cv_folds(fold_df)


def run_gnn_regressor_cv(
    train_df: pd.DataFrame,
    smiles_col: str,
    target_cols: list[str],
    *,
    architecture: str = "gin",
    config: Optional[BaselineCVConfig] = None,
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    hidden_dim: int = 64,
    num_layers: int = 3,
    gat_heads: int = 4,
    show_progress: bool = True,
    fit_show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """K-fold CV for :class:`~models.nn.pyg_regressor.PyGMoleculeRegressor`.

    ``architecture`` selects the PyG encoder (e.g. ``"gin"``, ``"gcn"``, ``"mpnn"``).
    ``fit_show_progress`` enables tqdm inside each fold's ``fit`` (default False).
    """
    cfg = config or BaselineCVConfig()
    work = prepare_regression_frame(train_df, smiles_col, target_cols)
    n_tasks = len(target_cols)
    kf = KFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=cfg.cv_random_state,
    )
    rows: list[dict[str, float | int]] = []
    fold_iter = enumerate(kf.split(np.arange(len(work))))
    if show_progress:
        fold_iter = tqdm(
            list(fold_iter),
            desc="gnn_cv",
            unit="fold",
        )
    for fold_id, (train_idx, val_idx) in fold_iter:
        train_part = work.iloc[train_idx]
        val_part = work.iloc[val_idx]
        train_ds = graph_regression_from_dataframe(
            train_part, smiles_col, target_cols
        )
        val_ds = graph_regression_from_dataframe(val_part, smiles_col, target_cols)
        model = PyGMoleculeRegressor(
            n_tasks=n_tasks,
            architecture=architecture,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gat_heads=gat_heads,
        )
        model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            val_dataset=val_ds,
            show_progress=fit_show_progress,
        )
        y_pred = model.predict(val_ds, show_progress=False)
        y_true = val_ds.y
        m = regression_metrics(y_true, y_pred)
        rows.append(
            {
                "fold": fold_id,
                **m,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            }
        )
    fold_df = pd.DataFrame(rows)
    return fold_df, summarize_cv_folds(fold_df)


def run_chemberta_regressor_cv(
    train_df: pd.DataFrame,
    smiles_col: str,
    target_cols: list[str],
    *,
    model_name_or_path: str = "DeepChem/ChemBERTa-77M-MLM",
    tokenizer_path: Optional[str] = "seyonec/PubChem10M_SMILES_BPE_60k",
    config: Optional[BaselineCVConfig] = None,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: Optional[int] = 256,
    show_progress: bool = True,
    fit_show_progress: bool = False,
    device: Optional[torch.device] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """K-fold CV using :class:`~models.pt.chemberta.ChembertaRegressor` (same loop as HF).

    Set ``fit_show_progress=True`` for per-epoch tqdm during each fold's ``fit``.
    """
    from models.pt.chemberta import ChembertaRegressor

    cfg = config or BaselineCVConfig()
    work = prepare_regression_frame(train_df, smiles_col, target_cols)
    n_tasks = len(target_cols)
    kf = KFold(
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
        random_state=cfg.cv_random_state,
    )
    rows: list[dict[str, float | int]] = []
    fold_iter = enumerate(kf.split(np.arange(len(work))))
    if show_progress:
        fold_iter = tqdm(
            list(fold_iter),
            desc="chemberta_cv",
            unit="fold",
        )
    for fold_id, (train_idx, val_idx) in fold_iter:
        train_part = work.iloc[train_idx]
        val_part = work.iloc[val_idx]
        train_ds = smiles_regression_from_dataframe(
            train_part, smiles_col, target_cols
        )
        val_ds = smiles_regression_from_dataframe(val_part, smiles_col, target_cols)
        model = ChembertaRegressor(
            model_name_or_path,
            n_tasks=n_tasks,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            device=device,
        )
        model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            val_dataset=val_ds,
            show_progress=fit_show_progress,
        )
        y_pred = model.predict(val_ds, show_progress=False)
        y_true = val_ds.y
        m = regression_metrics(y_true, y_pred)
        rows.append(
            {
                "fold": fold_id,
                **m,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            }
        )
    fold_df = pd.DataFrame(rows)
    return fold_df, summarize_cv_folds(fold_df)
