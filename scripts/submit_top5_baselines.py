#!/usr/bin/env python3
"""
Build **activity-track** submission CSVs for the top *k* (descriptor, regressor)
baselines from cross-validation, then fit each model on the **full** training set
and predict the blind test set.

Output files match :func:`submission.build_activity_submission` / the OpenADMET
`PXR-Challenge-Tutorial <https://github.com/OpenADMET/PXR-Challenge-Tutorial>`_
evaluation expectations (columns ``Molecule Name``, ``pEC50``).

**Prerequisite:** run baseline CV so ``outputs/baseline_cv_results.csv`` exists, e.g.::

  python src/score_data.py

**Upload:** submissions are uploaded manually through the challenge Space
`openadmet/pxr-challenge <https://huggingface.co/spaces/openadmet/pxr-challenge>`_
(see the tutorial repo for notebooks and evaluation).

Usage (from repo root)::

  python scripts/submit_top5_baselines.py
  python scripts/submit_top5_baselines.py --top-k 5 --out-dir outputs/pxr_submissions
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from baseline import BaselineCVConfig, fit_best_on_full_train  # noqa: E402
from features_data import build_descriptor_matrix  # noqa: E402
from load_data import get_test, get_train  # noqa: E402
from submission import build_activity_submission, validate_submission, write_submission  # noqa: E402

_CHALLENGE_SPACE = "https://huggingface.co/spaces/openadmet/pxr-challenge"
_TUTORIAL = "https://github.com/OpenADMET/PXR-Challenge-Tutorial"


def _safe_filename_part(s: str, max_len: int = 80) -> str:
    t = re.sub(r"[^\w.\-]+", "_", s, flags=re.ASCII)
    return t[:max_len].strip("_") or "run"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train top baseline CV rows on full data; write challenge submission CSVs."
    )
    p.add_argument(
        "--cv-results",
        type=Path,
        default=_ROOT / "outputs" / "baseline_cv_results.csv",
        help="CSV from baseline CV (sorted by mean_rmse)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "outputs" / "pxr_challenge_submissions",
        help="Directory for submission CSVs and manifest",
    )
    p.add_argument("--top-k", type=int, default=5, help="Number of top CV rows to export")
    p.add_argument(
        "--y-col",
        type=str,
        default="pEC50",
        help="Target column (must match CV that produced baseline_cv_results.csv)",
    )
    p.add_argument(
        "--model-random-state",
        type=int,
        default=0,
        help="Must match BaselineCVConfig.model_random_state used for baseline CV",
    )
    args = p.parse_args()

    if not args.cv_results.is_file():
        raise SystemExit(
            f"Missing {args.cv_results}. Run baseline CV first, e.g.:\n"
            f"  python {_ROOT / 'src' / 'score_data.py'}"
        )

    cv_df = pd.read_csv(args.cv_results)
    for col in ("descriptor", "model", "mean_rmse"):
        if col not in cv_df.columns:
            raise SystemExit(f"{args.cv_results} missing required column {col!r}")

    cv_df = cv_df.sort_values("mean_rmse", ascending=True).reset_index(drop=True)
    top = cv_df.head(args.top_k)

    train_df = get_train()
    test_df = get_test()
    mols_train = list(train_df["SMILES"].apply(Chem.MolFromSmiles))
    mols_test = list(test_df["SMILES"].apply(Chem.MolFromSmiles))

    cfg = BaselineCVConfig(y_col=args.y_col, model_random_state=args.model_random_state)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for rank, row in top.iterrows():
        desc = str(row["descriptor"])
        model_name = str(row["model"])
        pipe = fit_best_on_full_train(
            train_df,
            mols_train,
            desc,
            model_name,
            config=cfg,
        )
        X_test = build_descriptor_matrix(desc, mols_test).astype(np.float64)
        y_pred = pipe.predict(X_test)
        sub = build_activity_submission(test_df, y_pred, value_col=args.y_col)
        validate_submission(sub, test_df, value_col=args.y_col)

        stem = f"rank{rank + 1:02d}_{_safe_filename_part(desc)}_{_safe_filename_part(model_name)}"
        out_path = args.out_dir / f"{stem}_submission.csv"
        write_submission(out_path, sub, format="csv")

        manifest.append(
            {
                "rank": int(rank) + 1,
                "descriptor": desc,
                "model": model_name,
                "mean_rmse": float(row["mean_rmse"]),
                "submission_csv": str(out_path.resolve()),
            }
        )
        print(f"Wrote {out_path}  ({desc!r} + {model_name!r}, mean_rmse={row['mean_rmse']:.6f})")

    man_path = args.out_dir / "manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "challenge_space": _CHALLENGE_SPACE,
                "tutorial": _TUTORIAL,
                "cv_results": str(args.cv_results.resolve()),
                "y_col": args.y_col,
                "model_random_state": args.model_random_state,
                "runs": manifest,
            },
            f,
            indent=2,
        )
    print(f"\nManifest: {man_path}")
    print(
        f"\nUpload each *_submission.csv via the challenge Space:\n  {_CHALLENGE_SPACE}\n"
        f"Tutorial / evaluation layout:\n  {_TUTORIAL}"
    )


if __name__ == "__main__":
    main()
