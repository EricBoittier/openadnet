"""Uncertainty helpers: residual intervals, quantile models, and conformal QR."""

from .conformal_quantile import (
    ConformalizedQuantileRegressor,
    cross_validate_conformal_quantile,
)
from .plotting import (
    assay_ci_width,
    format_uncertainty_metrics_table,
    plot_emax_uncertainty_panel,
    plot_interval_width_comparison,
    plot_pred_vs_obs_with_intervals,
    uncertainty_comparison_metrics,
)
from .uncertainty import (
    assay_model_width_correlation,
    compare_interval_width_to_assay,
    fit_quantile_gradient_boosting,
    prediction_intervals_from_residuals,
    quantile_intervals_predict,
    residual_quantile_offsets,
)

__all__ = [
    "ConformalizedQuantileRegressor",
    "cross_validate_conformal_quantile",
    "assay_ci_width",
    "format_uncertainty_metrics_table",
    "plot_emax_uncertainty_panel",
    "plot_interval_width_comparison",
    "plot_pred_vs_obs_with_intervals",
    "uncertainty_comparison_metrics",
    "assay_model_width_correlation",
    "compare_interval_width_to_assay",
    "fit_quantile_gradient_boosting",
    "prediction_intervals_from_residuals",
    "quantile_intervals_predict",
    "residual_quantile_offsets",
]
