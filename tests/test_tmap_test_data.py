"""
TMAP layout from PXR challenge **test** (blinded) SMILES.

Loads ``pxr-challenge_TEST_BLINDED.csv`` via ``load_data.get_test()`` (Hugging Face Hub),
builds Morgan fingerprints → MinHash → LSH forest → 2D layout.

Requires network on first run (dataset download). Set ``SKIP_HF_INTEGRATION=1`` to skip.

Run from ``openadnet/``::

    python -m unittest tests.test_tmap_test_data -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "project"))

from tmap.tmap import LSHForest, LayoutResult, Minhash, layout_from_lsh_forest  # noqa: E402

# Same hub file as ``load_data.get_test()`` — avoid ``import load_data`` here because that
# module eagerly loads train+test on import.
_HF_REPO = "openadmet/pxr-challenge-train-test"
_HF_TEST_FILE = "pxr-challenge_TEST_BLINDED.csv"


def _load_test_df() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=_HF_REPO,
        filename=_HF_TEST_FILE,
        repo_type="dataset",
    )
    return pd.read_csv(path)

# Small subsample keeps CI/local runs fast; full test set is larger.
N_MOLECULES = 200
MINHASH_D = 128
FP_SIZE = 1024
MORGAN_RADIUS = 2
KNN_K = 10
KNN_KC = 10
FME_ITERS = 200


def _subsample_unique_smiles(df, n: int, smiles_col: str = "SMILES") -> list[str]:
    if smiles_col not in df.columns:
        raise AssertionError(
            f"Expected column {smiles_col!r}; got {list(df.columns)}"
        )
    seen: set[str] = set()
    out: list[str] = []
    for smi in df[smiles_col].astype(str):
        if not smi or smi in seen:
            continue
        seen.add(smi)
        out.append(smi)
        if len(out) >= n:
            break
    return out


@unittest.skipIf(
    os.environ.get("SKIP_HF_INTEGRATION", "").lower() in ("1", "true", "yes"),
    "SKIP_HF_INTEGRATION set — skipping Hugging Face test data load",
)
class TestTmapPxrTestData(unittest.TestCase):
    def test_tmap_layout_from_test_smiles(self) -> None:
        test_df = _load_test_df()

        smiles = _subsample_unique_smiles(test_df, N_MOLECULES)
        self.assertGreaterEqual(
            len(smiles),
            KNN_K + 1,
            "Need enough unique molecules for kNN graph",
        )

        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS, fpSize=FP_SIZE
        )
        fps: list[np.ndarray] = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fps.append(np.array(gen.GetFingerprint(mol), dtype=np.uint8))

        fps_array = np.array(fps)
        self.assertEqual(len(fps_array.shape), 2)
        self.assertGreater(fps_array.shape[0], KNN_K)

        mh = Minhash(d=MINHASH_D, seed=42)
        minhashes = mh.batch_from_binary_array(fps_array)

        lf = LSHForest(d=MINHASH_D, l=8, store=True)
        lf.batch_add(minhashes)
        lf.index()
        self.assertEqual(lf.size, fps_array.shape[0])

        result: LayoutResult = layout_from_lsh_forest(
            lf, k=KNN_K, kc=KNN_KC, fme_iterations=FME_ITERS
        )
        n = fps_array.shape[0]
        self.assertEqual(result.x.shape[0], n)
        self.assertEqual(result.y.shape[0], n)
        self.assertEqual(len(result.s), n - 1)
        self.assertEqual(len(result.t), n - 1)
        self.assertTrue(np.isfinite(result.x).all())
        self.assertTrue(np.isfinite(result.y).all())
        self.assertGreater(result.mst_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
