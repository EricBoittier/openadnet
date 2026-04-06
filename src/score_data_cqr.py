"""
Run conformalized quantile regression (CQR) baseline CV with the same outer folds
as :mod:`score_data`.

Uses :func:`baseline.run_baseline_cqr_cv`, which applies the same ``KFold`` as
:func:`baseline.run_baseline_cv` and, inside each training fold, splits fit vs
calibration for CQR. Point metrics (RMSE, MAE, R²) are from median predictions
on the held-out fold.

Writes ``outputs/baseline_cqr_cv_results.csv`` and ``outputs/baseline_cqr_cv_results.html``.
Does not overwrite the main README baseline table.
"""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from baseline import BaselineCVConfig, run_baseline_cqr_cv
from load_data import train
from reporting import write_baseline_cv_artifacts

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    mols = list(train["SMILES"].apply(Chem.MolFromSmiles))
    cfg = BaselineCVConfig(n_splits=5, cv_random_state=0, model_random_state=0)
    results = run_baseline_cqr_cv(
        train,
        mols,
        config=cfg,
        alpha=0.1,
        calibration_fraction=0.2,
        cqr_random_state=0,
    )
    csv_path, _ = write_baseline_cv_artifacts(
        results,
        _ROOT,
        csv_name="baseline_cqr_cv_results.csv",
        html_name="baseline_cqr_cv_results.html",
        update_readme=False,
    )
    print(results.to_string(index=False))
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
