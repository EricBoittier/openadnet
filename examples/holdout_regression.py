#!/usr/bin/env python3
"""
Single train / validation split (no K-fold): fit, print RMSE/MAE/R² on the holdout.

Run from repo root::

    PYTHONPATH=src python examples/holdout_regression.py --backend hf
    PYTHONPATH=src python examples/holdout_regression.py --backend gnn
    PYTHONPATH=src python examples/holdout_regression.py --backend gnn --gnn-architecture mpnn

    Requires: ``pip install openadnet[dl]``. For ``--backend hf``, Hub access for the model.
"""

from __future__ import annotations

import argparse

from load_data import train
from models.cv_dl import prepare_regression_frame, regression_metrics
from models.data import graph_regression_from_dataframe, smiles_regression_from_dataframe
from models.data.graph import train_val_split_graph
from models.data.transformer import train_val_split_smiles
from models.hf_regression import HuggingFaceRegressor
from models.nn.pyg_regressor import PyGMoleculeRegressor
from models.nn.registry import ARCHITECTURES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("hf", "gnn"), default="hf")
    p.add_argument(
        "--gnn-architecture",
        choices=list(ARCHITECTURES),
        default="gin",
        help="PyG encoder when --backend gnn",
    )
    p.add_argument(
        "--gat-heads",
        type=int,
        default=4,
        help="Attention heads for gat / attentivefp when --backend gnn",
    )
    p.add_argument("--hidden-dim", type=int, default=64, help="When --backend gnn")
    p.add_argument("--num-layers", type=int, default=3, help="When --backend gnn")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0, help="If >0, use only first N rows")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    df = train if args.max_rows <= 0 else train.head(args.max_rows)
    work = prepare_regression_frame(df, "SMILES", ["pEC50"])

    if args.backend == "hf":
        full_ds = smiles_regression_from_dataframe(work, "SMILES", ["pEC50"])
        tr_ds, va_ds = train_val_split_smiles(
            full_ds, val_fraction=args.val_fraction, random_state=args.seed
        )
        model = HuggingFaceRegressor(
            "hf-internal-testing/tiny-random-bert",
            n_tasks=1,
            max_length=64,
        )
        model.fit(
            tr_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=2e-5,
            val_dataset=va_ds,
            show_progress=True,
        )
        pred = model.predict(va_ds, show_progress=False)
        m = regression_metrics(va_ds.y, pred)
    else:
        full_ds = graph_regression_from_dataframe(work, "SMILES", ["pEC50"])
        tr_ds, va_ds = train_val_split_graph(
            full_ds, val_fraction=args.val_fraction, random_state=args.seed
        )
        model = PyGMoleculeRegressor(
            n_tasks=1,
            architecture=args.gnn_architecture,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            gat_heads=args.gat_heads,
        )
        model.fit(
            tr_ds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=1e-3,
            val_dataset=va_ds,
            show_progress=True,
        )
        pred = model.predict(va_ds, show_progress=False)
        m = regression_metrics(va_ds.y, pred)

    print("Validation metrics:", m)


if __name__ == "__main__":
    main()
