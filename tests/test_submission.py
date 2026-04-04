"""Tests for activity submission helpers (no Hugging Face download required)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from submission import (
    bootstrap_activity_metrics,
    build_activity_submission,
    score_activity_predictions_simple,
    validate_submission,
    write_submission,
)


class TestValidateSubmission(unittest.TestCase):
    def test_valid(self) -> None:
        test_df = pd.DataFrame(
            {"Molecule Name": ["A", "B", "C"], "SMILES": ["x", "y", "z"]}
        )
        sub = pd.DataFrame({"Molecule Name": ["B", "A", "C"], "pEC50": [1.0, 2.0, 3.0]})
        validate_submission(sub, test_df)

    def test_wrong_length(self) -> None:
        test_df = pd.DataFrame({"Molecule Name": ["A", "B"]})
        sub = pd.DataFrame({"Molecule Name": ["A"], "pEC50": [1.0]})
        with self.assertRaises(ValueError) as ctx:
            validate_submission(sub, test_df)
        self.assertIn("row count", str(ctx.exception))

    def test_wrong_ids(self) -> None:
        test_df = pd.DataFrame({"Molecule Name": ["A", "B"]})
        sub = pd.DataFrame({"Molecule Name": ["A", "X"], "pEC50": [1.0, 2.0]})
        with self.assertRaises(ValueError) as ctx:
            validate_submission(sub, test_df)
        self.assertIn("multiset", str(ctx.exception))

    def test_nan_prediction(self) -> None:
        test_df = pd.DataFrame({"Molecule Name": ["A", "B"]})
        sub = pd.DataFrame({"Molecule Name": ["A", "B"], "pEC50": [1.0, np.nan]})
        with self.assertRaises(ValueError) as ctx:
            validate_submission(sub, test_df)
        self.assertIn("NaN", str(ctx.exception))


class TestBuildSubmission(unittest.TestCase):
    def test_array_order(self) -> None:
        test_df = pd.DataFrame({"Molecule Name": ["m1", "m2"], "x": [1, 2]})
        sub = build_activity_submission(test_df, np.array([5.0, 6.0]))
        self.assertListEqual(sub["pEC50"].tolist(), [5.0, 6.0])
        self.assertListEqual(sub["Molecule Name"].tolist(), ["m1", "m2"])

    def test_series_reindex(self) -> None:
        test_df = pd.DataFrame({"Molecule Name": ["m1", "m2"]})
        s = pd.Series([6.0, 5.0], index=["m2", "m1"])
        sub = build_activity_submission(test_df, s)
        self.assertListEqual(sub["pEC50"].tolist(), [5.0, 6.0])

    def test_series_missing_raises(self) -> None:
        test_df = pd.DataFrame({"Molecule Name": ["m1", "m2"]})
        s = pd.Series([5.0], index=["m1"])
        with self.assertRaises(ValueError):
            build_activity_submission(test_df, s)


class TestScoreActivity(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        gt = pd.DataFrame({"Molecule Name": ["a", "b"], "pEC50": [4.0, 5.0]})
        pred = pd.DataFrame({"Molecule Name": ["a", "b"], "pEC50": [4.0, 5.0]})
        s = score_activity_predictions_simple(pred, gt)
        self.assertAlmostEqual(s["MAE"], 0.0, places=6)
        self.assertAlmostEqual(s["R2"], 1.0, places=6)

    def test_bootstrap_runs(self) -> None:
        y_t = np.array([1.0, 2.0, 3.0, 4.0])
        y_p = y_t + 0.1
        df = bootstrap_activity_metrics(
            y_p, y_t, n_bootstrap_samples=20, random_state=42
        )
        self.assertEqual(len(df), 20)
        self.assertIn("MAE", df.columns)
        self.assertEqual(df["Endpoint"].iloc[0], "pEC50")


class TestWriteSubmission(unittest.TestCase):
    def test_csv_roundtrip(self) -> None:
        sub = pd.DataFrame({"Molecule Name": ["a"], "pEC50": [3.14]})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sub.csv"
            write_submission(path, sub, format="csv")
            back = pd.read_csv(path)
            self.assertEqual(len(back), 1)
            self.assertAlmostEqual(back["pEC50"].iloc[0], 3.14)


if __name__ == "__main__":
    unittest.main()
