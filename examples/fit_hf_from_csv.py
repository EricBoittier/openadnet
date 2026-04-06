#!/usr/bin/env python3
"""
Train a Hugging Face regressor on **your own CSV** (SMILES + one or more numeric targets).

Example with a downloaded challenge file::

    PYTHONPATH=src python examples/fit_hf_from_csv.py \\
        --csv path/to/train.csv --smiles-col SMILES --targets pEC50 \\
        --model hf-internal-testing/tiny-random-bert --epochs 2 --save-dir outputs/my_model

Requires: ``pip install openadnet[dl]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.cv_dl import prepare_regression_frame, regression_metrics
from models.data import smiles_regression_from_dataframe
from models.data.transformer import train_val_split_smiles
from models.hf_regression import HuggingFaceRegressor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--targets", nargs="+", required=True)
    p.add_argument("--model", default="hf-internal-testing/tiny-random-bert")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=Path, default=None)
    args = p.parse_args()

    raw = pd.read_csv(args.csv)
    work = prepare_regression_frame(raw, args.smiles_col, list(args.targets))
    full_ds = smiles_regression_from_dataframe(work, args.smiles_col, list(args.targets))
    tr_ds, va_ds = train_val_split_smiles(
        full_ds, val_fraction=args.val_fraction, random_state=args.seed
    )

    model = HuggingFaceRegressor(
        args.model,
        n_tasks=len(args.targets),
        max_length=args.max_length,
    )
    model.fit(
        tr_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_dataset=va_ds,
        show_progress=True,
    )
    pred = model.predict(va_ds, show_progress=False)
    print("Validation:", regression_metrics(va_ds.y, pred))

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_dir)
        print("Saved to", args.save_dir)


if __name__ == "__main__":
    main()
