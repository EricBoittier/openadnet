#!/usr/bin/env python3
"""
Load a **saved** Hugging Face regression model from disk and run predictions.

Train + save once (example)::

    PYTHONPATH=src python -c "
    from pathlib import Path
    from load_data import train
    from models.cv_dl import prepare_regression_frame
    from models.data import smiles_regression_from_dataframe
    from models.hf_regression import HuggingFaceRegressor
    work = prepare_regression_frame(train.head(200), 'SMILES', ['pEC50'])
    ds = smiles_regression_from_dataframe(work, 'SMILES', ['pEC50'])
    m = HuggingFaceRegressor('hf-internal-testing/tiny-random-bert', n_tasks=1, max_length=64)
    m.fit(ds, epochs=1, batch_size=16, show_progress=False)
    out = Path('outputs/example_hf_reg')
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out)
    print('saved to', out)
    "

Then predict::

    PYTHONPATH=src python examples/load_hf_predict.py --model-dir outputs/example_hf_reg

Requires: ``pip install openadnet[dl]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from load_data import train
from models.cv_dl import prepare_regression_frame
from models.data import smiles_regression_from_dataframe
from models.hf_regression import HuggingFaceRegressor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory from HuggingFaceRegressor.save_pretrained",
    )
    p.add_argument("--n-rows", type=int, default=50, help="Predict on first N training rows")
    p.add_argument("--targets", nargs="+", default=["pEC50"])
    args = p.parse_args()

    sub = train.head(args.n_rows)
    work = prepare_regression_frame(sub, "SMILES", list(args.targets))
    ds = smiles_regression_from_dataframe(work, "SMILES", list(args.targets))

    model = HuggingFaceRegressor.from_pretrained(
        str(args.model_dir),
        n_tasks=len(args.targets),
        max_length=256,
    )
    pred = model.predict(ds, show_progress=False)
    print("y_true (first 5 rows):")
    print(ds.y[:5])
    print("\ny_pred (first 5 rows):")
    print(pred[:5])


if __name__ == "__main__":
    main()
