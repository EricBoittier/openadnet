#!/usr/bin/env python3
"""
Notebook-friendly helpers: graphs + baseline (cached) fingerprint rows on each node.

Per-molecule descriptors use the same pipeline as sklearn baselines
(:func:`features_data.build_descriptor_matrix` → joblib cache under ``OPENADNET_FP_CACHE``).

**Paste in a notebook** (add ``src`` to ``sys.path``; optional repo root if you import this file as ``examples....``)::

    from pathlib import Path
    import sys
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    sys.path.insert(0, str(root / "src"))

    from load_data import train
    from models.data import graph_regression_from_dataframe
    from models.nn.pyg_regressor import PyGMoleculeRegressor

    ds = graph_regression_from_dataframe(
        train.head(200), "SMILES", ["pEC50"], descriptor_name="morgan_r2_bits_512"
    )
    model = PyGMoleculeRegressor(
        1, descriptor_name="morgan_r2_bits_512", hidden_dim=48
    )
    model.fit(ds, epochs=2, batch_size=16, show_progress=True)

Or import the helpers below after ``sys.path.insert(0, str(root))`` (repo root, not only ``src``).

Requires: ``pip install openadnet[dl]``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from models.data import graph_regression_from_dataframe
from models.nn.pyg_regressor import PyGMoleculeRegressor


def make_graph_regression_dataset(
    df: pd.DataFrame,
    smiles_col: str,
    target_cols: Sequence[str],
    *,
    descriptor_name: str,
    drop_na_targets: bool = True,
):
    """Build :class:`~models.data.graph.GraphRegressionDataset` with broadcast descriptors."""
    return graph_regression_from_dataframe(
        df,
        smiles_col,
        list(target_cols),
        drop_na_targets=drop_na_targets,
        descriptor_name=descriptor_name,
    )


def make_pyg_regressor(
    n_tasks: int,
    *,
    descriptor_name: str,
    architecture: str = "gin",
    hidden_dim: int = 64,
    num_layers: int = 3,
    gat_heads: int = 4,
    device=None,
) -> PyGMoleculeRegressor:
    """Construct :class:`~models.nn.pyg_regressor.PyGMoleculeRegressor` with matching ``in_dim``."""
    return PyGMoleculeRegressor(
        n_tasks,
        architecture=architecture,
        descriptor_name=descriptor_name,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gat_heads=gat_heads,
        device=device,
    )


def main() -> None:
    """Tiny CLI sanity check (optional)."""
    from load_data import train

    small = train.head(50)
    name = "rdkit_phys_props"
    ds = make_graph_regression_dataset(small, "SMILES", ["pEC50"], descriptor_name=name)
    m = make_pyg_regressor(1, descriptor_name=name, hidden_dim=32, num_layers=2)
    m.fit(ds, epochs=1, batch_size=8, show_progress=False)
    p = m.predict(ds, show_progress=False)
    print("ok", p.shape)


if __name__ == "__main__":
    main()
