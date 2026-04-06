#!/usr/bin/env python3
"""
K-fold CV for the **SVR + LGBM + HGB + MPNN + GCN** ensemble (uniform average).

Fingerprints (``descriptor_name``) feed the three sklearn members; MPNN and GCN
use graph features only on the same folds as :func:`models.cv_dl.run_gnn_regressor_cv`.

Run from repo root::

    PYTHONPATH=src python scripts/cv_ensemble_fp_gnn.py --n-splits 5 --epochs 30
    PYTHONPATH=src python scripts/cv_ensemble_fp_gnn.py --cpu --epochs 20 --output outputs/ensemble_cv.csv

Requires ``pip install openadnet[dl]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from baseline import BaselineCVConfig
from load_data import train
from models.cv_dl import run_svr_lgbm_hgb_mpnn_gcn_ensemble_cv


def main() -> None:
    p = argparse.ArgumentParser(
        description="CV for SVR+LGBM+HGB+MPNN+GCN ensemble (fingerprints + PyG)."
    )
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--targets", nargs="+", default=["pEC50"])
    p.add_argument(
        "--descriptor-name",
        default="morgan_r1_count_1024",
        help="Fingerprint key for SVR/LGBM/HGB (features_data)",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--cv-seed", type=int, default=0)
    p.add_argument("--model-random-state", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--gat-heads", type=int, default=4)
    p.add_argument(
        "--verbose-fit",
        action="store_true",
        help="tqdm progress inside each GNN fit",
    )
    p.add_argument("--cpu", action="store_true", help="force CPU for PyG heads")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    cfg = BaselineCVConfig(
        n_splits=args.n_splits,
        shuffle=True,
        cv_random_state=args.cv_seed,
        y_col=args.targets[0],
    )
    fold_df, summary = run_svr_lgbm_hgb_mpnn_gcn_ensemble_cv(
        train,
        smiles_col=args.smiles_col,
        target_cols=list(args.targets),
        descriptor_name=args.descriptor_name,
        config=cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gat_heads=args.gat_heads,
        show_progress=True,
        fit_show_progress=args.verbose_fit,
        device=torch.device("cpu") if args.cpu else None,
        model_random_state=args.model_random_state,
    )
    print("Per-fold metrics:")
    print(fold_df.to_string(index=False))
    print("\nSummary (mean ± std):")
    print(summary.to_string())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fold_df.to_csv(args.output, index=False)
        summary.to_csv(args.output.with_suffix(".summary.csv"))
        print(f"\nWrote {args.output} and {args.output.with_suffix('.summary.csv')}")


if __name__ == "__main__":
    main()
