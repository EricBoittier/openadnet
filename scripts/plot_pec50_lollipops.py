#!/usr/bin/env python3
"""
Render ordered pEC50 lollipops with assay CI error bars (train table from HF hub).

Usage (from ``openadnet/``):

  python scripts/plot_pec50_lollipops.py
  python scripts/plot_pec50_lollipops.py -o outputs/pEC50_lollipops.png --descending
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from load_data import get_train
from viz import plot_pec50_lollipops


def main() -> None:
    p = argparse.ArgumentParser(description="pEC50 lollipop plot with CI error bars")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/pEC50_lollipops.png"),
        help="Output image path",
    )
    p.add_argument(
        "--descending",
        action="store_true",
        help="Order from highest to lowest pEC50 (default is lowest → highest)",
    )
    args = p.parse_args()
    ascending = not args.descending

    train = get_train()
    plot_pec50_lollipops(train, ascending=ascending)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
