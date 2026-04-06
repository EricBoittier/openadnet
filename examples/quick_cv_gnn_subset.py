#!/usr/bin/env python3
"""
Fast PyG GNN CV on a **subset** of rows (all architectures in ``models.nn.registry.ARCHITECTURES``).

Run from repo root::

    PYTHONPATH=src python examples/quick_cv_gnn_subset.py
    PYTHONPATH=src python examples/quick_cv_gnn_subset.py --architecture gcn
    PYTHONPATH=src python examples/quick_cv_gnn_subset.py --architecture mpnn --epochs 3

Requires: ``pip install openadnet[dl]``.
"""

from __future__ import annotations

import argparse

from baseline import BaselineCVConfig
from load_data import train
from models.cv_dl import run_gnn_regressor_cv
from models.nn.registry import ARCHITECTURES

N_ROWS_DEFAULT = 300


def main() -> None:
    p = argparse.ArgumentParser(description="Quick CV on a train subset (PyG).")
    p.add_argument("--architecture", choices=list(ARCHITECTURES), default="gin")
    p.add_argument("--gat-heads", type=int, default=4)
    p.add_argument("--n-rows", type=int, default=N_ROWS_DEFAULT)
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument(
        "--descriptor-name",
        type=str,
        default=None,
        help="Optional: broadcast baseline fingerprint onto each node",
    )
    args = p.parse_args()

    small = train.head(args.n_rows).copy()
    cfg = BaselineCVConfig(
        n_splits=args.n_splits,
        shuffle=True,
        cv_random_state=42,
        y_col="pEC50",
    )
    fold_df, summary = run_gnn_regressor_cv(
        small,
        smiles_col="SMILES",
        target_cols=["pEC50"],
        architecture=args.architecture,
        descriptor_name=args.descriptor_name,
        config=cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gat_heads=args.gat_heads,
        show_progress=True,
    )
    print(fold_df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
