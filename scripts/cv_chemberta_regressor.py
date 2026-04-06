#!/usr/bin/env python3
"""
K-fold cross-validation for :class:`~models.pt.chemberta.ChembertaRegressor`.

Same CV settings as ``baseline.BaselineCVConfig``. Defaults match the ChemBERTa +
PubChem SMILES BPE setup used in the literature.

Run from the repo root::

    PYTHONPATH=src python scripts/cv_chemberta_regressor.py \\
        --epochs 1 --n-splits 3 --model DeepChem/ChemBERTa-77M-MLM

Requires ``pip install openadnet[dl]`` and Hub access for tokenizer/weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from baseline import BaselineCVConfig
from load_data import train
from models.cv_dl import run_chemberta_regressor_cv
from models.pt.chemberta import ChembertaRegressor
from models.data import smiles_regression_from_dataframe


def main() -> None:
    p = argparse.ArgumentParser(description="K-fold CV for ChemBERTa regression.")
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--targets", nargs="+", default=["pEC50"])
    p.add_argument(
        "--model",
        default="DeepChem/ChemBERTa-77M-MLM",
        help="Encoder checkpoint (HF id or path)",
    )
    p.add_argument(
        "--tokenizer",
        default="seyonec/PubChem10M_SMILES_BPE_60k",
        help="Tokenizer id/path (ignored if --tokenizer-from-model)",
    )
    p.add_argument(
        "--tokenizer-from-model",
        action="store_true",
        help="Load tokenizer from the same repo as --model",
    )
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--cv-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (recommended if you see CUDA device-side assert errors)",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, refit on all filtered rows and save model+tokenizer here",
    )
    args = p.parse_args()

    tok: str | None
    if args.tokenizer_from_model:
        tok = None
    else:
        tok = args.tokenizer or None

    cfg = BaselineCVConfig(
        n_splits=args.n_splits,
        shuffle=True,
        cv_random_state=args.cv_seed,
        y_col=args.targets[0],
    )
    fold_df, summary = run_chemberta_regressor_cv(
        train,
        smiles_col=args.smiles_col,
        target_cols=list(args.targets),
        model_name_or_path=args.model,
        tokenizer_path=tok,
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
        model = ChembertaRegressor(
            args.model,
            n_tasks=len(args.targets),
            tokenizer_path=tok,
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
