#!/usr/bin/env python3
"""
K-fold cross-validation for :class:`~models.nn.pyg_regressor.PyGMoleculeRegressor`
(PyG encoders: gin, gcn, gat, graphconv, mpnn, attentivefp).

Run from the repo root::

    PYTHONPATH=src python scripts/cv_gnn_regressor.py --epochs 2 --n-splits 3
    PYTHONPATH=src python scripts/cv_gnn_regressor.py --architecture mpnn --epochs 2

Requires ``pip install openadnet[dl]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from baseline import BaselineCVConfig
from load_data import train
from models.cv_dl import run_gnn_regressor_cv
from models.data import graph_regression_from_dataframe
from models.nn.pyg_regressor import PyGMoleculeRegressor
from models.nn.registry import ARCHITECTURES


def main() -> None:
    p = argparse.ArgumentParser(description="K-fold CV for PyG molecular GNN regression.")
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--targets", nargs="+", default=["pEC50"])
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--cv-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--architecture",
        choices=list(ARCHITECTURES),
        default="gin",
        help="PyG encoder (see models.nn.registry.ARCHITECTURES)",
    )
    p.add_argument(
        "--gat-heads",
        type=int,
        default=4,
        help="Attention heads for gat and attentivefp",
    )
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument(
        "--verbose-fit",
        action="store_true",
        help="Show tqdm epoch/batch bars during each fold's training",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, refit on all filtered rows and save weights here (gnn_regression.pt)",
    )
    args = p.parse_args()

    cfg = BaselineCVConfig(
        n_splits=args.n_splits,
        shuffle=True,
        cv_random_state=args.cv_seed,
        y_col=args.targets[0],
    )
    fold_df, summary = run_gnn_regressor_cv(
        train,
        smiles_col=args.smiles_col,
        target_cols=list(args.targets),
        architecture=args.architecture,
        config=cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gat_heads=args.gat_heads,
        show_progress=True,
        fit_show_progress=args.verbose_fit,
    )
    print("Per-fold metrics:")
    print(fold_df.to_string(index=False))
    print("\nSummary (mean ± std):")
    print(summary.to_string())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")

    if args.save_dir is not None:
        from models.cv_dl import prepare_regression_frame

        work = prepare_regression_frame(train, args.smiles_col, list(args.targets))
        full_ds = graph_regression_from_dataframe(work, args.smiles_col, list(args.targets))
        model = PyGMoleculeRegressor(
            n_tasks=len(args.targets),
            architecture=args.architecture,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            gat_heads=args.gat_heads,
        )
        model.fit(
            full_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            show_progress=True,
        )
        args.save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_dir)
        print(f"Saved full-data model to {args.save_dir}")


if __name__ == "__main__":
    main()
