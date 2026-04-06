"""
Reusable plots for prediction intervals vs assay-reported uncertainties.

Compare conformal / model interval widths to assay CI width and standard errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


# --- Column names (PXR challenge training table) ---------------------------------

PEC50 = "pEC50"
PEC50_CI_LOWER = "pEC50_ci.lower (-log10(molarity))"
PEC50_CI_UPPER = "pEC50_ci.upper (-log10(molarity))"
PEC50_STD = "pEC50_std.error (-log10(molarity))"

EMAX = "Emax_estimate (log2FC vs. baseline)"
EMAX_CI_LOWER = "Emax_ci.lower (log2FC vs. baseline)"
EMAX_CI_UPPER = "Emax_ci.upper (log2FC vs. baseline)"
EMAX_STD = "Emax_std.error (log2FC vs. baseline)"

EMAX_VS_PC = "Emax.vs.pos.ctrl_estimate (dimensionless)"
EMAX_VS_PC_CI_LOWER = "Emax.vs.pos.ctrl_ci.lower (dimensionless)"
EMAX_VS_PC_CI_UPPER = "Emax.vs.pos.ctrl_ci.upper (dimensionless)"
EMAX_VS_PC_STD = "Emax.vs.pos.ctrl_std.error (dimensionless)"


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(np.asarray(a, dtype=float))
    return m


def assay_ci_width(
    df: pd.DataFrame,
    lower_col: str = PEC50_CI_LOWER,
    upper_col: str = PEC50_CI_UPPER,
) -> np.ndarray:
    """Per-row CI width (upper − lower); NaN if either bound missing."""
    lo = pd.to_numeric(df[lower_col], errors="coerce")
    hi = pd.to_numeric(df[upper_col], errors="coerce")
    return (hi - lo).to_numpy(dtype=float)


def plot_pred_vs_obs_with_intervals(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    assay_ci_lower: np.ndarray | None = None,
    assay_ci_upper: np.ndarray | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    xlabel: str = "Observed pEC50",
    ylabel: str = "Predicted pEC50 (median)",
    point_size: float = 14.0,
    point_alpha: float = 0.55,
    line_alpha: float = 0.35,
    capsize: float = 1.2,
    identity: bool = True,
) -> Axes:
    """
    Scatter observed vs predicted with asymmetric error bars for **model** intervals
    on the prediction (vertical). Optionally add horizontal error bars from **assay**
    CI on the observed value.

    Error bar convention: model uses ``[lower, upper]`` around ``y_pred``; assay uses
    ``[assay_ci_lower, assay_ci_upper]`` around ``y_obs``.
    """
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    y_lo_err = np.clip(y_pred - lower, 0.0, None)
    y_hi_err = np.clip(upper - y_pred, 0.0, None)
    yerr = np.vstack([y_lo_err, y_hi_err])

    if assay_ci_lower is not None and assay_ci_upper is not None:
        acl = np.asarray(assay_ci_lower, dtype=float).ravel()
        acu = np.asarray(assay_ci_upper, dtype=float).ravel()
        x_lo_err = np.clip(y_obs - acl, 0.0, None)
        x_hi_err = np.clip(acu - y_obs, 0.0, None)
        xerr = np.vstack([x_lo_err, x_hi_err])
        m = _finite_mask(y_obs, y_pred, lower, upper, acl, acu)
        ax.errorbar(
            y_obs[m],
            y_pred[m],
            xerr=xerr[:, m],
            yerr=yerr[:, m],
            fmt="none",
            ecolor="C0",
            elinewidth=0.8,
            capsize=capsize,
            alpha=line_alpha,
            zorder=1,
        )
    else:
        m = _finite_mask(y_obs, y_pred, lower, upper)
        ax.errorbar(
            y_obs[m],
            y_pred[m],
            yerr=yerr[:, m],
            fmt="none",
            ecolor="C0",
            elinewidth=0.8,
            capsize=capsize,
            alpha=line_alpha,
            zorder=1,
        )

    ax.scatter(
        y_obs[m],
        y_pred[m],
        s=point_size,
        alpha=point_alpha,
        c="C0",
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
    )

    if identity:
        lo = float(np.nanmin([y_obs[m].min(), y_pred[m].min()]))
        hi = float(np.nanmax([y_obs[m].max(), y_pred[m].max()]))
        pad = 0.02 * (hi - lo + 1e-9)
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            "k--",
            lw=1,
            alpha=0.7,
            label="y = ŷ",
        )
        ax.legend(loc="upper left")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_interval_width_comparison(
    model_width: np.ndarray,
    assay_ci_width: np.ndarray,
    *,
    assay_std: np.ndarray | None = None,
    ax: Axes | None = None,
    axes: tuple[Axes, Axes] | None = None,
    title: str | None = None,
    xlabel_ci: str = "Assay CI width (upper − lower)",
    xlabel_std: str = "Assay std error",
    ylabel: str = "Model interval width",
) -> Axes | tuple[Axes, Axes]:
    """
    Scatter comparing CQR interval width to assay CI width (same units as the target).
    If ``assay_std`` is given, also plot model width vs std (second panel if ``ax`` is None).
    """
    mw = np.asarray(model_width, dtype=float).ravel()
    aw = np.asarray(assay_ci_width, dtype=float).ravel()
    m = _finite_mask(mw, aw)
    mw_c, aw_c = mw[m], aw[m]

    if assay_std is None:
        if ax is None:
            _, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(
            aw_c,
            mw_c,
            alpha=0.45,
            s=18,
            c="C0",
            edgecolors="white",
            linewidths=0.3,
        )
        hi = float(max(np.nanmax(aw_c), np.nanmax(mw_c)))
        lo = float(min(np.nanmin(aw_c), np.nanmin(mw_c)))
        pad = 0.02 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, alpha=0.6)
        ax.set_xlabel(xlabel_ci)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        return ax

    st = np.asarray(assay_std, dtype=float).ravel()
    if axes is not None:
        ax_ci, ax_st = axes
    elif ax is not None:
        raise ValueError("Pass ``axes=(ax_ci, ax_std)`` when using assay_std, or omit ``ax``.")
    else:
        _, (ax_ci, ax_st) = plt.subplots(1, 2, figsize=(11, 4.8))

    ax_ci.scatter(
        aw_c,
        mw_c,
        alpha=0.45,
        s=18,
        c="C0",
        edgecolors="white",
        linewidths=0.3,
    )
    hi = float(max(np.nanmax(aw_c), np.nanmax(mw_c)))
    lo = float(min(np.nanmin(aw_c), np.nanmin(mw_c)))
    pad = 0.02 * (hi - lo + 1e-9)
    ax_ci.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, alpha=0.6)
    ax_ci.set_xlabel(xlabel_ci)
    ax_ci.set_ylabel(ylabel)
    ax_ci.set_title("vs assay CI width")

    m2 = _finite_mask(mw, st)
    ax_st.scatter(
        st[m2],
        mw[m2],
        alpha=0.45,
        s=18,
        c="C1",
        edgecolors="white",
        linewidths=0.3,
    )
    hi2 = float(max(np.nanmax(st[m2]), np.nanmax(mw[m2])))
    lo2 = float(min(np.nanmin(st[m2]), np.nanmin(mw[m2])))
    pad2 = 0.02 * (hi2 - lo2 + 1e-9)
    ax_st.plot([lo2 - pad2, hi2 + pad2], [lo2 - pad2, hi2 + pad2], "k--", lw=1, alpha=0.6)
    ax_st.set_xlabel(xlabel_std)
    ax_st.set_ylabel(ylabel)
    ax_st.set_title("vs assay std error")
    if title:
        plt.suptitle(title, y=1.02)
    return ax_ci, ax_st


def plot_emax_uncertainty_panel(
    df: pd.DataFrame,
    model_lower: np.ndarray,
    model_upper: np.ndarray,
    *,
    axes: tuple[Axes, Axes, Axes] | None = None,
    suptitle: str | None = None,
) -> tuple[Axes, Axes, Axes]:
    """
    Three panels: CQR **pEC50** interval width vs (1) pEC50 assay CI — same units, diagonal;
    (2–3) same model width vs **Emax** and **Emax vs pos ctrl** assay CI widths — different
    units, no diagonal; Pearson *r* in subtitle (cross-endpoint association).
    """
    mw = np.asarray(model_upper, dtype=float) - np.asarray(model_lower, dtype=float)

    if axes is None:
        _, axes_arr = plt.subplots(1, 3, figsize=(12.5, 4))
        ax0, ax1, ax2 = axes_arr[0], axes_arr[1], axes_arr[2]
    else:
        ax0, ax1, ax2 = axes

    def _pec50(ax: Axes) -> None:
        aw = assay_ci_width(df, lower_col=PEC50_CI_LOWER, upper_col=PEC50_CI_UPPER)
        m = _finite_mask(mw, aw)
        ax.scatter(aw[m], mw[m], alpha=0.45, s=14, c="C0", edgecolors="white", linewidths=0.25)
        hi = float(max(np.nanmax(aw[m]), np.nanmax(mw[m])))
        lo = float(min(np.nanmin(aw[m]), np.nanmin(mw[m])))
        pad = 0.02 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.9, alpha=0.55)
        ax.set_xlabel("pEC50 assay CI width")
        ax.set_ylabel("CQR pEC50 interval width")
        ax.set_title("pEC50 (same units)")

    def _cross(ax: Axes, ci_lo: str, ci_hi: str, title: str) -> None:
        aw = assay_ci_width(df, lower_col=ci_lo, upper_col=ci_hi)
        m = _finite_mask(mw, aw)
        if m.sum() >= 3:
            r = float(np.corrcoef(mw[m], aw[m])[0, 1])
            ax.set_title(f"{title}\n(r = {r:.3f})")
        else:
            ax.set_title(title)
        ax.scatter(aw[m], mw[m], alpha=0.45, s=14, c="C2", edgecolors="white", linewidths=0.25)
        ax.set_xlabel("Assay CI width")
        ax.set_ylabel("CQR pEC50 interval width")

    _pec50(ax0)
    _cross(ax1, EMAX_CI_LOWER, EMAX_CI_UPPER, "Emax")
    _cross(ax2, EMAX_VS_PC_CI_LOWER, EMAX_VS_PC_CI_UPPER, "Emax vs pos ctrl")

    if suptitle:
        ax0.figure.suptitle(suptitle, y=1.05)
    return ax0, ax1, ax2


def uncertainty_comparison_metrics(
    model_width: np.ndarray,
    assay_ci_width: np.ndarray,
    assay_std: np.ndarray | None = None,
) -> dict[str, float]:
    """Pearson correlation between model width and assay CI width / std (finite rows only)."""
    mw = np.asarray(model_width, dtype=float).ravel()
    aw = np.asarray(assay_ci_width, dtype=float).ravel()
    m = _finite_mask(mw, aw)
    out: dict[str, float] = {}
    if m.sum() >= 3:
        out["r_model_width_vs_assay_ci_width"] = float(
            np.corrcoef(mw[m], aw[m])[0, 1]
        )
    else:
        out["r_model_width_vs_assay_ci_width"] = float("nan")

    if assay_std is not None:
        st = np.asarray(assay_std, dtype=float).ravel()
        m2 = _finite_mask(mw, st)
        if m2.sum() >= 3:
            out["r_model_width_vs_assay_std"] = float(
                np.corrcoef(mw[m2], st[m2])[0, 1]
            )
        else:
            out["r_model_width_vs_assay_std"] = float("nan")
    return out


def format_uncertainty_metrics_table(metrics: dict[str, float]) -> str:
    lines = []
    for k, v in sorted(metrics.items()):
        if np.isfinite(v):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: nan")
    return "\n".join(lines) if lines else "(no metrics)"
