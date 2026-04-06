#!/usr/bin/env python3
"""
Fast Hugging Face CV on a **subset** of rows (for debugging pipelines).

Run from repo root::

    PYTHONPATH=src python examples/quick_cv_hf_subset.py

Requires: ``pip install openadnet[dl]``, Hub access for the tiny test model.
"""

from __future__ import annotations

from baseline import BaselineCVConfig
from load_data import train
from models.cv_dl import run_hf_regressor_cv

# First N rows only — increase for more stable metrics
N_ROWS = 256
MODEL = "hf-internal-testing/tiny-random-bert"


def main() -> None:
    small = train.head(N_ROWS).copy()
    cfg = BaselineCVConfig(n_splits=3, shuffle=True, cv_random_state=42, y_col="pEC50")
    fold_df, summary = run_hf_regressor_cv(
        small,
        smiles_col="SMILES",
        target_cols=["pEC50"],
        model_name_or_path=MODEL,
        tokenizer_name_or_path=None,
        config=cfg,
        epochs=1,
        batch_size=16,
        learning_rate=2e-5,
        max_length=64,
        show_progress=True,
    )
    print(fold_df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
