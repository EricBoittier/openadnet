#!/usr/bin/env python3
"""
Fast PyG GNN CV on a **subset** of rows.

Run from repo root::

    PYTHONPATH=src python examples/quick_cv_gnn_subset.py

Requires: ``pip install openadnet[dl]``.
"""

from __future__ import annotations

from baseline import BaselineCVConfig
from load_data import train
from models.cv_dl import run_gnn_regressor_cv

N_ROWS = 300


def main() -> None:
    small = train.head(N_ROWS).copy()
    cfg = BaselineCVConfig(n_splits=3, shuffle=True, cv_random_state=42, y_col="pEC50")
    fold_df, summary = run_gnn_regressor_cv(
        small,
        smiles_col="SMILES",
        target_cols=["pEC50"],
        config=cfg,
        epochs=2,
        batch_size=32,
        learning_rate=1e-3,
        hidden_dim=48,
        num_layers=2,
        show_progress=True,
    )
    print(fold_df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
