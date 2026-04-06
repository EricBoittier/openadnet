"""Smoke tests for uncertainty.plotting."""

import numpy as np
import pandas as pd

from uncertainty.plotting import (
    PEC50_CI_LOWER,
    PEC50_CI_UPPER,
    PEC50_STD,
    assay_ci_width,
    uncertainty_comparison_metrics,
)


def test_assay_ci_width_and_metrics():
    df = pd.DataFrame(
        {
            PEC50_CI_LOWER: [5.0, np.nan, 6.0],
            PEC50_CI_UPPER: [6.0, 7.0, 8.0],
            PEC50_STD: [0.2, 0.3, 0.4],
        }
    )
    w = assay_ci_width(df)
    assert w.shape == (3,)
    assert np.isnan(w[1])

    mw = np.array([1.0, 2.0, 3.0])
    m = uncertainty_comparison_metrics(mw, w, df[PEC50_STD].to_numpy())
    assert "r_model_width_vs_assay_ci_width" in m
    assert "r_model_width_vs_assay_std" in m
