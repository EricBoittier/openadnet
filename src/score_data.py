"""
Example entrypoint: load PXR training data and run baseline CV grid.

First import of ``load_data`` may download CSVs from the Hub; later runs use the
Hugging Face disk cache (and in-memory LRU cache). See ``load_data`` docstring.

Writes ``outputs/baseline_cv_results.csv``, ``outputs/baseline_cv_results.html``,
and updates the HTML table in ``README.md`` between marker comments.
"""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from baseline import BaselineCVConfig, run_baseline_cv
from load_data import train
from reporting import write_baseline_cv_artifacts

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    mols = list(train["SMILES"].apply(Chem.MolFromSmiles))
    cfg = BaselineCVConfig(n_splits=5, cv_random_state=0, model_random_state=0)
    results = run_baseline_cv(train, mols, config=cfg)
    csv_path, readme_path = write_baseline_cv_artifacts(results, _ROOT)
    print(results.to_string(index=False))
    print(f"\nWrote {csv_path}")
    if readme_path is not None:
        print(f"Updated {readme_path}")


if __name__ == "__main__":
    main()
