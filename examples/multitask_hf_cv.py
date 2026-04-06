#!/usr/bin/env python3
"""
K-fold CV with **two regression targets** (multi-task head).

Uses ``pEC50`` and ``Emax_estimate (log2FC vs. baseline)`` from the PXR training table.
Rows missing either target are dropped by :func:`models.cv_dl.prepare_regression_frame`.

Run::

    PYTHONPATH=src python examples/multitask_hf_cv.py

Requires: ``pip install openadnet[dl]``.
"""

from __future__ import annotations

from baseline import BaselineCVConfig
from load_data import train
from models.cv_dl import run_hf_regressor_cv

TARGETS = [
    "pEC50",
    "Emax_estimate (log2FC vs. baseline)",
]
MODEL = "hf-internal-testing/tiny-random-bert"


def main() -> None:
    cfg = BaselineCVConfig(n_splits=3, shuffle=True, cv_random_state=0, y_col=TARGETS[0])
    fold_df, summary = run_hf_regressor_cv(
        train,
        smiles_col="SMILES",
        target_cols=TARGETS,
        model_name_or_path=MODEL,
        tokenizer_name_or_path=None,
        config=cfg,
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
        max_length=96,
        show_progress=True,
    )
    print(fold_df.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
