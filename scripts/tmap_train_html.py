#!/usr/bin/env python3
"""
Build an **annotated TMAP** (SVG) from PXR **train** SMILES and write HTML + optional standalone SVG.

Uses the same pipeline as ``tests/test_tmap_annotated.py`` (Voronoi/scaffold layers optional,
Murcko scaffold cards, direct SVG — no matplotlib): see ``project/tmap/annotated_svg.py``.

The default HTML view includes a **magnifier** panel (fixed top-right) that follows the
pointer in SVG coordinates and reports the **nearest molecule** by known node positions.

Usage (from ``openadnet/``)::

  python scripts/tmap_train_html.py -o outputs/tmap_train.html
  python scripts/tmap_train_html.py --max-molecules 800 -o outputs/tmap_sample.html
  python scripts/tmap_train_html.py --no-magnifier -o outputs/tmap_train_plain.html
  python scripts/tmap_train_html.py --hide-scaffolds -o outputs/tmap_no_scaffolds.html

Magnification defaults come from ``--mag-zoom`` (1–100×); the interactive HTML
(``tmap_viewer_html``) adds a **logarithmic Zoom** slider (linear drag → 1×–100×) and a
**selected-structure** panel (SMILES + 2D draw via SmilesDrawer CDN). Use
``--no-structure-viewer`` for offline / no-CDN use.

Train **pEC50** (or ``--activity-col``) colors nodes (viridis) and draws a **colorbar** on the
map and a matching scale in the side panel; ``--no-activity-color`` disables coloring.

The interactive HTML also includes a **filterable molecule table** (name, value, SMILES): type at
least two characters to search, then click a row to **center the magnifier** on that point.

For large maps, use ``--no-magnifier-lens`` so Chrome does not keep a second copy of the graph in the
magnifier (big memory win). Use ``--no-data-table`` if you only need the map.

Use ``--embedding-compare`` to append **PCA / UMAP / t-SNE** panels (same fingerprints, standardized;
same viridis scale as TMAP when activity coloring is on). t-SNE is skipped when
``n > --embedding-tsne-max`` (default 5000) to keep runtime reasonable.

Requires Hugging Face Hub access on first run (dataset download cache).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "project"))

from tmap.annotated_svg import build_annotated_tmap_svg  # noqa: E402
from fp_embedding_compare import build_embedding_comparison_html  # noqa: E402
from tmap_viewer_html import (  # noqa: E402
    merge_smiles_into_nodes,
    write_tmap_html_interactive,
    write_tmap_html_simple,
)

_HF_REPO = "openadmet/pxr-challenge-train-test"
_HF_TRAIN_FILE = "pxr-challenge_TRAIN.csv"

# Defaults aligned with ``tests/test_tmap_annotated.py`` (annotated SVG path)
DEFAULT_MINHASH_D = 128
DEFAULT_FP_SIZE = 2048
DEFAULT_MORGAN_RADIUS = 3
DEFAULT_KNN_K = 10
DEFAULT_KNN_KC = 5
DEFAULT_FME_ITERS = 150
DEFAULT_SVG_SIZE = 8000
DEFAULT_REPEL_ITERS = 200
DEFAULT_MAG_ZOOM = 4.0


def _load_train_df() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=_HF_REPO,
        filename=_HF_TRAIN_FILE,
        repo_type="dataset",
    )
    return pd.read_csv(path)


def _collect_smiles_names_activity(
    df: pd.DataFrame,
    *,
    smiles_col: str,
    name_col: str,
    activity_col: str | None,
    max_molecules: int | None,
    unique_only: bool,
) -> tuple[list[str], list[str], list[float | None]]:
    if smiles_col not in df.columns:
        raise SystemExit(f"Column {smiles_col!r} not found; columns: {list(df.columns)}")
    seen: set[str] = set()
    smiles_out: list[str] = []
    names_out: list[str] = []
    activity_out: list[float | None] = []
    has_name = name_col in df.columns
    has_activity = bool(activity_col) and activity_col in df.columns
    for _, row in df.iterrows():
        smi = str(row[smiles_col])
        if not smi or smi == "nan":
            continue
        if unique_only and smi in seen:
            continue
        if unique_only:
            seen.add(smi)
        smiles_out.append(smi)
        if has_name:
            names_out.append(str(row[name_col]))
        else:
            names_out.append(f"#{len(smiles_out) - 1}")
        if has_activity:
            raw = row[activity_col]
            num = pd.to_numeric(raw, errors="coerce")
            activity_out.append(float(num) if pd.notna(num) else None)
        else:
            activity_out.append(None)
        if max_molecules is not None and len(smiles_out) >= max_molecules:
            break
    if len(smiles_out) < DEFAULT_KNN_K + 1:
        raise SystemExit(
            f"Need at least {DEFAULT_KNN_K + 1} molecules for kNN; got {len(smiles_out)}."
        )
    return smiles_out, names_out, activity_out


def _build_mols_and_fps(
    smiles: list[str],
    labels: list[str],
    activity: list[float | None],
    *,
    fp_size: int,
    radius: int,
) -> tuple[list, np.ndarray, list[str], list[str], list[float | None]]:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    mols: list = []
    fps: list[np.ndarray] = []
    kept_labels: list[str] = []
    kept_smiles: list[str] = []
    kept_activity: list[float | None] = []
    for smi, lab, act in zip(smiles, labels, activity, strict=True):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        kept_labels.append(lab)
        kept_smiles.append(smi)
        kept_activity.append(act)
        fps.append(np.array(gen.GetFingerprint(mol), dtype=np.uint8))
    if len(mols) < DEFAULT_KNN_K + 1:
        raise SystemExit(
            f"Too few valid SMILES after RDKit parse; got {len(mols)} molecules."
        )
    return mols, np.array(fps), kept_labels, kept_smiles, kept_activity


def main() -> None:
    p = argparse.ArgumentParser(
        description="PXR train TMAP → annotated SVG + HTML (see test_tmap_annotated.py)."
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_ROOT / "outputs" / "tmap_train.html",
        help="Output HTML path (default: outputs/tmap_train.html)",
    )
    p.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        metavar="N",
        help="Cap number of molecules after SMILES collection (default: all)",
    )
    p.add_argument(
        "--all-rows",
        action="store_true",
        help="Include duplicate SMILES as separate nodes (default: unique SMILES, row order)",
    )
    p.add_argument(
        "--smiles-col",
        default="SMILES",
        help="SMILES column name (default: SMILES)",
    )
    p.add_argument(
        "--name-col",
        default="Molecule Name",
        help="Label column for magnifier (default: Molecule Name)",
    )
    p.add_argument(
        "--activity-col",
        default="pEC50",
        metavar="COL",
        help="Numeric column for node color + colorbar (default: pEC50; use pIC50 if your table has it)",
    )
    p.add_argument(
        "--no-activity-color",
        action="store_true",
        help="Ignore activity column: uniform node color, no colorbar",
    )
    p.add_argument("--minhash-d", type=int, default=DEFAULT_MINHASH_D)
    p.add_argument("--fp-size", type=int, default=DEFAULT_FP_SIZE)
    p.add_argument("--morgan-radius", type=int, default=DEFAULT_MORGAN_RADIUS)
    p.add_argument("--knn-k", type=int, default=DEFAULT_KNN_K)
    p.add_argument("--knn-kc", type=int, default=DEFAULT_KNN_KC)
    p.add_argument("--fme-iterations", type=int, default=DEFAULT_FME_ITERS)
    p.add_argument("--svg-size", type=int, default=DEFAULT_SVG_SIZE)
    p.add_argument("--repel-iters", type=int, default=DEFAULT_REPEL_ITERS)
    p.add_argument(
        "--show-voronoi",
        action="store_true",
        help="Draw Voronoi cells and merged scaffold hulls (slower, heavier SVG)",
    )
    p.add_argument(
        "--show-mol-structures",
        action="store_true",
        help="Draw per-cell molecule structures for singletons (heavy)",
    )
    p.add_argument(
        "--hide-scaffolds",
        action="store_true",
        help="Omit Murcko scaffold cards, labels, and merged hull outlines (with --show-voronoi); tree + dots only",
    )
    p.add_argument(
        "--min-scaffold-group",
        type=int,
        default=20,
        help="Minimum Murcko group size for scaffold cards (default: 20)",
    )
    p.add_argument(
        "--scaffold-type",
        choices=("murcko", "generic", "brics"),
        default="murcko",
    )
    p.add_argument(
        "--no-standalone-svg",
        action="store_true",
        help="Do not write a separate .svg next to the .html",
    )
    p.add_argument(
        "--no-magnifier",
        action="store_true",
        help="Plain HTML wrapper without magnifier / nearest-molecule panel",
    )
    p.add_argument(
        "--mag-zoom",
        type=float,
        default=DEFAULT_MAG_ZOOM,
        help=(
            f"Initial magnifier zoom 1–100× (clamped); browser uses a log slider over the same "
            f"range (default: {DEFAULT_MAG_ZOOM})"
        ),
    )
    p.add_argument(
        "--no-structure-viewer",
        action="store_true",
        help="Omit SMILES + 2D structure panel (no SmilesDrawer CDN)",
    )
    p.add_argument(
        "--no-data-table",
        action="store_true",
        help="Omit the searchable molecule table (saves a bit of JS/DOM work)",
    )
    p.add_argument(
        "--no-magnifier-lens",
        action="store_true",
        help=(
            "Do not duplicate the graph inside the magnifier (<use href=#tmap-graph/>). "
            "Strongly recommended for large maps — Chrome otherwise keeps two copies in memory."
        ),
    )
    p.add_argument(
        "--embedding-compare",
        action="store_true",
        help=(
            "Add comparable PCA / UMAP / t-SNE 2D plots (standardized Morgan FP; same colors as TMAP). "
            "Requires umap-learn. t-SNE is omitted when n exceeds --embedding-tsne-max."
        ),
    )
    p.add_argument(
        "--embedding-tsne-max",
        type=int,
        default=5000,
        metavar="N",
        help="Max molecules for which t-SNE is computed (default: 5000; larger sets are slow)",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Less console output")
    args = p.parse_args()

    print("Loading train CSV from Hugging Face Hub …")
    train_df = _load_train_df()
    act_col = None if args.no_activity_color else args.activity_col
    smiles, mol_names, activity_row = _collect_smiles_names_activity(
        train_df,
        smiles_col=args.smiles_col,
        name_col=args.name_col,
        activity_col=act_col,
        max_molecules=args.max_molecules,
        unique_only=not args.all_rows,
    )
    n_req = len(smiles)
    print(f"SMILES collected: {n_req} (unique_only={not args.all_rows})")

    print("Morgan fingerprints + RDKit mols …")
    mols, fps_array, labels, kept_smiles, kept_activity = _build_mols_and_fps(
        smiles,
        mol_names,
        activity_row,
        fp_size=args.fp_size,
        radius=args.morgan_radius,
    )
    n = len(mols)
    if n != n_req:
        print(
            f"Note: using {n} molecules after dropping invalid SMILES (requested {n_req}).",
            file=sys.stderr,
        )

    title = f"TMAP – PXR train ({n} molecules)"
    page_title = f"TMAP – PXR train ({n} molecules)"

    print("Annotated TMAP (MinHash → LSH → layout → SVG) …")
    node_vals = None
    node_lbl = args.activity_col
    if not args.no_activity_color:
        arr = np.array(
            [np.nan if v is None else float(v) for v in kept_activity],
            dtype=float,
        )
        if np.any(np.isfinite(arr)):
            node_vals = arr
        else:
            print(
                "No finite activity values; drawing nodes without color scale.",
                file=sys.stderr,
            )

    svg_content, nodes, colorbar_meta = build_annotated_tmap_svg(
        mols,
        fps_array,
        title=title,
        minhash_d=args.minhash_d,
        knn_k=args.knn_k,
        knn_kc=args.knn_kc,
        fme_iterations=args.fme_iterations,
        svg_size=args.svg_size,
        repel_iters=args.repel_iters,
        show_voronoi=args.show_voronoi,
        show_mol_structures=args.show_mol_structures,
        show_scaffolds=not args.hide_scaffolds,
        min_scaffold_group=args.min_scaffold_group,
        scaffold_type=args.scaffold_type,
        verbose=not args.quiet,
        labels=labels,
        node_values=node_vals,
        node_value_label=node_lbl,
    )

    embed_html: str | None = None
    if args.embedding_compare:
        if not args.quiet:
            print(
                "Computing PCA / UMAP / t-SNE embedding comparison (standardized fingerprints) …",
                flush=True,
            )
        embed_html = build_embedding_comparison_html(
            fps_array,
            node_vals,
            value_label=node_lbl,
            tsne_max_points=max(50, args.embedding_tsne_max),
        )

    if args.no_magnifier:
        write_tmap_html_simple(
            args.output,
            svg_content,
            page_title=page_title,
            extra_html=embed_html,
        )
    else:
        nodes_full = merge_smiles_into_nodes(nodes, kept_smiles)
        write_tmap_html_interactive(
            args.output,
            svg_content,
            page_title=page_title,
            nodes=nodes_full,
            mag_zoom=args.mag_zoom,
            show_structure_viewer=not args.no_structure_viewer,
            colorbar=colorbar_meta,
            show_data_table=not args.no_data_table,
            magnifier_clone_graph=not args.no_magnifier_lens,
            extra_html_before_script=embed_html,
        )
    print(f"Wrote {args.output}")

    if not args.no_standalone_svg:
        svg_path = args.output.with_suffix(".svg")
        svg_path.write_text(svg_content, encoding="utf-8")
        print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
