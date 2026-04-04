"""Smoke tests for PCA / UMAP / t-SNE embedding comparison plots."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from fp_embedding_compare import build_embedding_comparison_html, compute_2d_embeddings


class TestFpEmbeddingCompare(unittest.TestCase):
    def test_compute_2d_embeddings_small(self) -> None:
        rng = np.random.default_rng(0)
        fps = rng.integers(0, 2, size=(45, 32), dtype=np.uint8)
        layouts, pca = compute_2d_embeddings(fps, random_state=0, tsne_max_points=5000)
        self.assertEqual(layouts.pca.shape, (45, 2))
        self.assertEqual(layouts.umap.shape, (45, 2))
        self.assertIsNotNone(layouts.tsne)
        assert layouts.tsne is not None
        self.assertEqual(layouts.tsne.shape, (45, 2))
        self.assertEqual(pca.explained_variance_ratio_.shape, (2,))

    def test_build_embedding_html_includes_svg(self) -> None:
        rng = np.random.default_rng(1)
        fps = rng.integers(0, 2, size=(55, 48), dtype=np.uint8)
        vals = rng.standard_normal(55)
        html = build_embedding_comparison_html(
            fps, vals, value_label="test", tsne_max_points=5000
        )
        self.assertIn("<svg", html)
        self.assertIn("tmap-embed-compare", html)


if __name__ == "__main__":
    unittest.main()
