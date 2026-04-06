#!/usr/bin/env python3
"""
K-fold cross-validation for :class:`~models.hf_regression.HuggingFaceRegressor`.

Uses the same split semantics as ``baseline.BaselineCVConfig`` (``KFold`` with
``n_splits``, ``shuffle``, ``cv_random_state``).

Run from the repo root::

    PYTHONPATH=src python scripts/cv_hf_regressor.py \\
        --model hf-internal-testing/tiny-random-bert --epochs 1 --n-splits 3

Requires ``pip install openadnet[dl]`` and Hugging Face Hub access for the chosen model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from baseline import BaselineCVConfig
from load_data import train
from models.cv_dl import run_hf_regressor_cv
from models.hf_regression import HuggingFaceRegressor
from models.data import smiles_regression_from_dataframe


def main() -> None:
    p = argparse.ArgumentParser(description="K-fold CV for Hugging Face SMILES regression.")
    p.add_argument("--smiles-col", default="SMILES", help="SMILES column name")
    p.add_argument(
        "--targets",
        nargs="+",
        default=["pEC50"],
        help="Regression target column(s)",
    )
    p.add_argument(
        "--model",
        default="hf-internal-testing/tiny-random-bert",
        help="HF model id or local path (sequence classification, regression head)",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="Optional separate tokenizer id/path (default: same as --model)",
    )
    p.add_argument("--n-splits", type=int, default=3, help="KFold splits (same as baseline CV)")
    p.add_argument("--cv-seed", type=int, default=0, help="KFold random state")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (recommended if you see CUDA device-side assert errors)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for per-fold metrics",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, refit on all filtered rows and save model+tokenizer here",
    )
    args = p.parse_args()

    cfg = BaselineCVConfig(
        n_splits=args.n_splits,
        shuffle=True,
        cv_random_state=args.cv_seed,
        y_col=args.targets[0],
    )
    fold_df, summary = run_hf_regressor_cv(
        train,
        smiles_col=args.smiles_col,
        target_cols=list(args.targets),
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        config=cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        show_progress=True,
        device=torch.device("cpu") if args.cpu else None,
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
        full_ds = smiles_regression_from_dataframe(work, args.smiles_col, list(args.targets))
        model = HuggingFaceRegressor(
            args.model,
            n_tasks=len(args.targets),
            tokenizer_name_or_path=args.tokenizer,
            max_length=args.max_length,
            device=torch.device("cpu") if args.cpu else None,
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
