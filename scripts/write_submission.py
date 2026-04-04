#!/usr/bin/env python3
"""
Validate activity predictions and write a challenge submission CSV (or Parquet).

Predictions file options:
  - Two columns: Molecule Name, pEC50
  - One numeric column only: values must align with test row order (same as Hugging Face test CSV).

Usage (from ``openadnet/``):

  python scripts/write_submission.py -p outputs/preds.csv -o outputs/submission.csv
  python scripts/write_submission.py -p preds_one_col.csv --single-column-order -o sub.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from load_data import get_test
from submission import validate_submission, write_submission


def _load_predictions(path: Path, single_column_order: bool, test_df: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(path)
    if single_column_order:
        num = pred.select_dtypes(include=["number"])
        if num.shape[1] != 1:
            raise ValueError(
                "With --single-column-order, predictions CSV must have exactly one numeric column."
            )
        col = num.columns[0]
        if len(pred) != len(test_df):
            raise ValueError(
                f"Predictions rows {len(pred)} != test rows {len(test_df)}."
            )
        return pd.DataFrame(
            {"Molecule Name": test_df["Molecule Name"].values, "pEC50": pred[col].astype(float).values}
        )
    for c in ("Molecule Name", "pEC50"):
        if c not in pred.columns:
            raise ValueError(f"Predictions CSV must contain column {c!r} (or use --single-column-order).")
    return pred[["Molecule Name", "pEC50"]].copy()


def main() -> None:
    p = argparse.ArgumentParser(description="Validate predictions and write submission file")
    p.add_argument(
        "-p",
        "--predictions",
        type=Path,
        required=True,
        help="CSV of predictions (see script docstring)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path (.csv or .parquet)",
    )
    p.add_argument(
        "--test",
        type=Path,
        default=None,
        help="Optional test CSV path (default: load from Hugging Face via load_data.get_test)",
    )
    p.add_argument(
        "--single-column-order",
        action="store_true",
        help="Predictions file is one numeric column in blind test row order",
    )
    args = p.parse_args()

    if args.test is not None:
        test_df = pd.read_csv(args.test)
    else:
        test_df = get_test()

    sub = _load_predictions(args.predictions, args.single_column_order, test_df)
    validate_submission(sub, test_df)

    fmt: str = "parquet" if args.output.suffix.lower() == ".parquet" else "csv"
    write_submission(args.output, sub, format=fmt)
    print(f"Wrote {args.output} ({len(sub)} rows, format={fmt})")


if __name__ == "__main__":
    main()
