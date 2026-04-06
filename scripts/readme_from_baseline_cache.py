#!/usr/bin/env python3
"""
Rebuild ``README.md`` baseline table (and ``outputs/baseline_cv_results.*``) from
``outputs/baseline_cv_cache.json``.

By default this **only** reads the JSON cache and fails if any
(descriptor, model) pair is missing for the current training set. Use
``--fill-missing`` to run :func:`baseline.run_baseline_cv` for missing pairs
(same as ``src/score_data.py``), then write artifacts.

Usage (from repo root)::

    PYTHONPATH=src python scripts/readme_from_baseline_cache.py
    PYTHONPATH=src python scripts/readme_from_baseline_cache.py --fill-missing
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rdkit import Chem

from baseline import (
    BaselineCVConfig,
    dataframe_from_cv_cache,
    run_baseline_cv,
)
from load_data import train
from reporting import write_baseline_cv_artifacts

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Update README baseline CV table from baseline_cv_cache.json."
    )
    p.add_argument(
        "--fill-missing",
        action="store_true",
        help="Run CV for cache misses (updates JSON), then write README/CSV/HTML.",
    )
    args = p.parse_args()

    mols = list(train["SMILES"].apply(Chem.MolFromSmiles))
    cfg = BaselineCVConfig(n_splits=5, cv_random_state=0, model_random_state=0)

    if args.fill_missing:
        results = run_baseline_cv(train, mols, config=cfg)
    else:
        results = dataframe_from_cv_cache(train, mols, config=cfg)

    csv_path, readme_path = write_baseline_cv_artifacts(results, _ROOT)
    print(results.to_string(index=False))
    print(f"\nWrote {csv_path}")
    if readme_path is not None:
        print(f"Updated {readme_path}")


if __name__ == "__main__":
    main()
