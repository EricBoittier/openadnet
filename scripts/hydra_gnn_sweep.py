#!/usr/bin/env python3
"""
Hydra entry point for **PyG GNN K-fold CV** (``run_gnn_regressor_cv``).

Install: ``pip install 'openadnet[dl,hydra]'`` (or ``uv sync --extra dl --extra hydra``).

Run from the **repository root** with ``src`` on ``PYTHONPATH``::

    PYTHONPATH=src python scripts/hydra_gnn_sweep.py

**Single run** (defaults in ``conf/gnn_sweep/config.yaml``)::

    PYTHONPATH=src python scripts/hydra_gnn_sweep.py model.architecture=mpnn train.epochs=50

**Multirun sweeps** (grid over comma-separated values)::

    PYTHONPATH=src python scripts/hydra_gnn_sweep.py --multirun \\
        model.architecture=gin,gat,mpnn \\
        model.hidden_dim=64,128 \\
        train.learning_rate=1e-3,3e-4

**Descriptor on nodes**::

    PYTHONPATH=src python scripts/hydra_gnn_sweep.py \\
        model.descriptor_name=morgan_r2_bits_512 model.architecture=mpnn

Outputs per job under ``outputs/hydra_gnn/`` (see ``hydra.run.dir`` / ``hydra.sweep.dir`` in config).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

try:
    from hydra import main as hydra_main
    from omegaconf import DictConfig, OmegaConf
except ImportError as _e:  # pragma: no cover
    sys.stderr.write(
        "hydra-core is required. Install with: pip install 'openadnet[hydra]' "
        "or uv sync --extra hydra\n"
    )
    raise SystemExit(1) from _e


def _descriptor_name(value: Any) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in ("null", "none", "~"):
            return None
        return s
    if isinstance(value, (list, tuple)):
        out = [str(x).strip() for x in value if str(x).strip()]
        return out if out else None
    return str(value)


@hydra_main(version_base=None, config_path="../conf/gnn_sweep", config_name="config")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig

    from baseline import BaselineCVConfig
    from load_data import train
    from models.cv_dl import run_gnn_regressor_cv

    df = train.copy()
    max_rows = cfg.data.max_rows
    if max_rows is not None:
        df = df.head(int(max_rows))

    targets = list(cfg.data.targets)
    desc = _descriptor_name(cfg.model.descriptor_name)

    bc = BaselineCVConfig(
        n_splits=int(cfg.cv.n_splits),
        shuffle=True,
        cv_random_state=int(cfg.cv.cv_seed),
        y_col=str(targets[0]),
    )

    fold_df, summary = run_gnn_regressor_cv(
        df,
        smiles_col=str(cfg.data.smiles_col),
        target_cols=targets,
        architecture=str(cfg.model.architecture),
        descriptor_name=desc,
        config=bc,
        epochs=int(cfg.train.epochs),
        batch_size=int(cfg.train.batch_size),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        gat_heads=int(cfg.model.gat_heads),
        show_progress=bool(cfg.run.show_progress),
        fit_show_progress=bool(cfg.run.fit_show_progress),
    )

    print("\nPer-fold metrics:")
    print(fold_df.to_string(index=False))
    print("\nSummary (mean ± std):")
    print(summary.to_string())

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.output.save_csv):
        fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
        summary.to_csv(out_dir / "summary.csv")
    if bool(cfg.output.save_config):
        with open(out_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
