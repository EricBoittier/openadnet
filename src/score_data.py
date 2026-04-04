"""
Example entrypoint: load PXR training data and run baseline CV grid.

Requires network access to fetch HuggingFace CSVs on first import of `load_data`.
"""

from __future__ import annotations

from rdkit import Chem

from baseline import BaselineCVConfig, run_baseline_cv
from load_data import train


def main() -> None:
    mols = list(train["SMILES"].apply(Chem.MolFromSmiles))
    cfg = BaselineCVConfig(n_splits=5, cv_random_state=0, model_random_state=0)
    results = run_baseline_cv(train, mols, config=cfg)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
