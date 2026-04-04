from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec


def plot_target_distribution(
    df: pd.DataFrame,
    col: str = "pEC50",
    *,
    ax: plt.Axes | None = None,
    bins: int = 40,
    **kwargs: Any,
) -> plt.Axes:
    """Histogram of the regression target."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    s = df[col].dropna()
    ax.hist(s, bins=bins, density=True, alpha=0.75, edgecolor="black", **kwargs)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {col}")
    return ax


def plot_assay_uncertainty_vs_target(
    df: pd.DataFrame,
    target_col: str = "pEC50",
    std_col: str = "pEC50_std.error (-log10(molarity))",
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter of assay-reported std error vs pEC50 (EDA)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sub = df[[target_col, std_col]].dropna()
    ax.scatter(sub[target_col], sub[std_col], alpha=0.25, s=8)
    ax.set_xlabel(target_col)
    ax.set_ylabel(std_col)
    ax.set_title("Assay uncertainty vs target")
    return ax


def plot_ci_width_vs_target(
    df: pd.DataFrame,
    target_col: str = "pEC50",
    lower_col: str = "pEC50_ci.lower (-log10(molarity))",
    upper_col: str = "pEC50_ci.upper (-log10(molarity))",
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    width = df[upper_col] - df[lower_col]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sub = pd.DataFrame({target_col: df[target_col], "width": width}).dropna()
    ax.scatter(sub[target_col], sub["width"], alpha=0.25, s=8)
    ax.set_xlabel(target_col)
    ax.set_ylabel("CI width")
    ax.set_title("Assay CI width vs target")
    return ax


def plot_pec50_lollipops(
    df: pd.DataFrame,
    *,
    value_col: str = "pEC50",
    lower_col: str = "pEC50_ci.lower (-log10(molarity))",
    upper_col: str = "pEC50_ci.upper (-log10(molarity))",
    label_col: str | None = "Molecule Name",
    ascending: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    ytick_max: int = 60,
    color: str = "steelblue",
    ecolor: str = "0.35",
) -> plt.Axes:
    """
    Horizontal lollipop chart: rows ordered by ``value_col``, point at the estimate,
    asymmetric error bars from assay CI lower/upper (``-log10(molarity)`` scale).

    For large ``n``, y-axis labels are omitted unless ``n <= ytick_max``.
    """
    cols = [value_col, lower_col, upper_col]
    if label_col is not None and label_col in df.columns:
        use = df[[label_col, *cols]].dropna()
    else:
        use = df[cols].dropna()
        label_col = None

    use = use.sort_values(value_col, ascending=ascending).reset_index(drop=True)
    n = len(use)
    if n == 0:
        raise ValueError("No rows with finite pEC50 and CI bounds.")

    y = np.arange(n, dtype=float)
    x = use[value_col].to_numpy()
    lo = use[lower_col].to_numpy()
    hi = use[upper_col].to_numpy()
    xerr = np.vstack([np.clip(x - lo, 0, np.inf), np.clip(hi - x, 0, np.inf)])

    if ax is None:
        if figsize is None:
            w = 8.0
            h = max(5.0, min(0.14 * n, 48.0))
            figsize = (w, h)
        _, ax = plt.subplots(figsize=figsize)

    # Stem from zero (or data min) for lollipop look; CI shown separately
    xmin = float(np.nanmin(np.concatenate([lo, x])))
    pad = 0.05 * (float(np.nanmax(hi)) - xmin + 1e-9)
    x0 = min(0.0, xmin - pad) if xmin >= 0 else xmin - pad

    ax.hlines(y, x0, x, color=color, alpha=0.45, linewidth=0.8, zorder=1)
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        color=color,
        ecolor=ecolor,
        elinewidth=0.9,
        capsize=1.5,
        markersize=3.5,
        zorder=2,
    )
    ax.set_xlabel("pEC50 (−log₁₀ molarity)")
    ax.set_ylabel("")
    ax.set_title("pEC50 (ordered) with assay CI")
    ax.grid(axis="x", alpha=0.25)

    if label_col is not None and n <= ytick_max:
        labels = use[label_col].astype(str).tolist()
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_yticks([])

    ax.invert_yaxis()
    plt.tight_layout()
    return ax


def plot_eda_distributions_and_correlations(
    df: pd.DataFrame,
    *,
    exclude_cols: tuple[str, ...] | None = None,
    hist_bins: int = 30,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Histograms for each numeric column and a Pearson correlation heatmap.

    Excludes non-numeric columns (e.g. ``SMILES``, ``Molecule Name``, ``mol``) by default.
    """
    default_exclude = (
        "mol",
        "SMILES",
        "Molecule Name",
        "OCNT Batch",
        "Split",
    )
    exclude = set(default_exclude)
    if exclude_cols:
        exclude.update(exclude_cols)

    num = df.select_dtypes(include=[np.number]).copy()
    for c in exclude:
        if c in num.columns:
            num = num.drop(columns=[c])

    num = num.loc[:, num.notna().any()]
    if num.shape[1] == 0:
        raise ValueError("No numeric columns left after exclusions.")

    n_cols = num.shape[1]
    n_hist_rows = max(1, int(np.ceil(n_cols / 4)))
    if figsize is None:
        figsize = (14.0, 4.0 + 2.6 * n_hist_rows)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.15])

    ax_corr = fig.add_subplot(gs[0])
    corr = num.corr(numeric_only=True)
    im = ax_corr.imshow(corr.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax_corr.set_xticks(np.arange(len(corr.columns)))
    ax_corr.set_yticks(np.arange(len(corr.columns)))
    short_labels = [
        c.replace(" (-log10(molarity))", "")
        .replace(" (log2FC vs. baseline)", "")
        .replace(" (dimensionless)", "")[:32]
        for c in corr.columns
    ]
    ax_corr.set_xticklabels(short_labels, rotation=55, ha="right", fontsize=7)
    ax_corr.set_yticklabels(short_labels, fontsize=7)
    ax_corr.set_title("Pearson correlation (numeric columns)")
    fig.colorbar(im, ax=ax_corr, fraction=0.03, pad=0.02)

    gsh = gs[1].subgridspec(n_hist_rows, 4, hspace=0.4, wspace=0.28)
    for i, col in enumerate(num.columns):
        r, c = divmod(i, 4)
        ax_h = fig.add_subplot(gsh[r, c])
        s = num[col].dropna()
        ax_h.hist(s, bins=hist_bins, density=True, alpha=0.8, edgecolor="black", linewidth=0.3)
        title = (
            col.replace(" (-log10(molarity))", "")
            .replace(" (log2FC vs. baseline)", "")
            .replace(" (dimensionless)", "")
        )
        if len(title) > 28:
            title = title[:25] + "…"
        ax_h.set_title(title, fontsize=8)
        ax_h.tick_params(axis="both", labelsize=6)

    fig.suptitle("EDA: distributions and correlations", fontsize=12, y=1.02)
    return fig


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """CV predictions: y vs ŷ with identity line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, label=label)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted (CV OOF)")
    ax.set_title("Predicted vs actual")
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_residuals_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    res = y_true - y_pred
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, res, alpha=0.3, s=10)
    ax.axhline(0.0, color="k", linestyle="--", lw=1)
    ax.set_xlabel("Predicted (CV OOF)")
    ax.set_ylabel("Residual (y − ŷ)")
    ax.set_title("Residuals vs predicted")
    return ax


def plot_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    bins: int = 40,
) -> plt.Axes:
    res = y_true - y_pred
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(res, bins=bins, density=True, alpha=0.75, edgecolor="black")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution (CV OOF)")
    return ax


def plot_model_comparison(
    results_df: pd.DataFrame,
    *,
    metric: str = "mean_rmse",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Axes:
    """Bar chart of CV metric by descriptor × model (wide table)."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    pivot = results_df.pivot_table(
        index="descriptor", columns="model", values=metric, aggfunc="first"
    )
    pivot.plot(kind="bar", ax=ax, rot=45)
    ax.set_ylabel(metric)
    ax.set_title(f"Baseline CV comparison ({metric})")
    ax.legend(title="model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel("descriptor")
    plt.tight_layout()
    return ax


def set_notebook_style() -> None:
    """Match PXR tutorial notebook defaults (``whitegrid``, ``notebook`` context)."""
    sns.set_style("whitegrid")
    sns.set_context("notebook")


def plot_pec50_distribution_kde(
    df: pd.DataFrame,
    *,
    col: str = "pEC50",
    ax: plt.Axes | None = None,
    bins: int = 40,
    color: str = "steelblue",
    figsize: tuple[float, float] = (8.0, 4.0),
) -> plt.Axes:
    """Histogram + KDE of ``col`` for rows with non-null ``col`` (tutorial-style)."""
    pec50_df = df.dropna(subset=[col]).reset_index(drop=True)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if len(pec50_df) == 0:
        ax.set_title(f"{col}: no data")
        return ax
    sns.histplot(pec50_df[col], bins=bins, kde=True, ax=ax, color=color)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(f"pEC50 distribution (N = {len(pec50_df)}/{len(df)})")
    return ax


def plot_emax_pair_histograms(
    df: pd.DataFrame,
    *,
    emax_col: str = "Emax_estimate (log2FC vs. baseline)",
    emax_ctrl_col: str = "Emax.vs.pos.ctrl_estimate (dimensionless)",
    axes: tuple[plt.Axes, plt.Axes] | None = None,
    figsize: tuple[float, float] = (12.0, 4.0),
    bins: int = 40,
) -> tuple[plt.Axes, plt.Axes]:
    """Side-by-side KDE histograms for Emax and Emax vs. positive control (tutorial-style)."""
    if axes is None:
        _, ax_pair = plt.subplots(1, 2, figsize=figsize)
        ax0, ax1 = ax_pair[0], ax_pair[1]
    else:
        ax0, ax1 = axes

    for ax, col, title, xlab in (
        (
            ax0,
            emax_col,
            f"Emax (N = {df[emax_col].notna().sum()})",
            r"Emax (log$_2$FC vs. baseline)",
        ),
        (
            ax1,
            emax_ctrl_col,
            f"Emax vs. pos. ctrl (N = {df[emax_ctrl_col].notna().sum()})",
            "Emax vs. pos. ctrl (dimensionless)",
        ),
    ):
        s = df[col].dropna()
        if len(s) == 0:
            ax.set_title(f"{col}: no data")
            continue
        sns.histplot(s, bins=bins, kde=True, ax=ax, color="coral" if ax is ax0 else "darkorange")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Count")
        ax.set_title(title)

    return ax0, ax1


def _merge_counter_assay_plot_df(
    train_df: pd.DataFrame,
    train_counter_df: pd.DataFrame,
    *,
    id_col: str = "Molecule Name",
    primary_pec50: str = "pEC50",
    counter_pec50: str = "pEC50",
) -> pd.DataFrame:
    c = train_counter_df.rename(columns={counter_pec50: "counter_pEC50"})
    return c.merge(
        train_df[[id_col, primary_pec50]],
        on=id_col,
        how="inner",
    )


def plot_counter_assay_triage(
    train_df: pd.DataFrame | None = None,
    train_counter_df: pd.DataFrame | None = None,
    *,
    merged: pd.DataFrame | None = None,
    id_col: str = "Molecule Name",
    primary_pec50: str = "pEC50",
    counter_pec50: str = "pEC50",
    potency_threshold: float = 6.0,
    selectivity_margin: float = 1.5,
    plot_floor: float = 1.0,
    xlim: tuple[float, float] = (1.0, 8.0),
    ylim: tuple[float, float] = (1.0, 8.0),
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7.0, 7.0),
) -> plt.Axes:
    """
    Primary vs counter-assay pEC50 scatter with selectivity band (PXR tutorial).

    Pass either ``merged`` (columns ``pEC50``, ``counter_pEC50``) or both
    ``train_df`` and ``train_counter_df`` (merged on ``id_col``).
    """
    if merged is None:
        if train_df is None or train_counter_df is None:
            raise ValueError("Provide ``merged`` or both ``train_df`` and ``train_counter_df``.")
        counter_plot_df = _merge_counter_assay_plot_df(
            train_df,
            train_counter_df,
            id_col=id_col,
            primary_pec50=primary_pec50,
            counter_pec50=counter_pec50,
        )
    else:
        counter_plot_df = merged.copy()

    if "counter_pEC50" not in counter_plot_df.columns:
        raise ValueError("Merged data must contain column 'counter_pEC50'.")
    if primary_pec50 not in counter_plot_df.columns:
        raise ValueError(f"Merged data must contain primary column {primary_pec50!r}.")

    counter_plot_df["selectivity_delta"] = (
        counter_plot_df[primary_pec50] - counter_plot_df["counter_pEC50"]
    )
    counter_plot_df["is_selective"] = (counter_plot_df[primary_pec50] >= potency_threshold) & (
        counter_plot_df["selectivity_delta"] >= selectivity_margin
    )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    x_range = np.linspace(0, 10, 100)
    ax.fill_between(
        x_range,
        plot_floor,
        x_range - selectivity_margin,
        where=(x_range >= potency_threshold) & (x_range - selectivity_margin > plot_floor),
        color="green",
        alpha=0.15,
        label="Potent and selective region",
    )
    sns.scatterplot(
        data=counter_plot_df,
        x=primary_pec50,
        y="counter_pEC50",
        hue="is_selective",
        palette={False: "darkorange", True: "darkgreen"},
        ax=ax,
        alpha=0.65,
        s=28,
    )
    ax.plot(x_range, x_range, color="black", linestyle="--", alpha=0.4, label="No selectivity")
    ax.plot(
        x_range,
        x_range - selectivity_margin,
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"{selectivity_margin:g} log unit selectivity margin",
    )
    ax.axvline(potency_threshold, color="navy", linestyle="-.", alpha=0.7, label=f"Primary pEC50 = {potency_threshold:g}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Primary-assay pEC50", fontsize=12)
    ax.set_ylabel("Counter-assay pEC50", fontsize=12)
    ax.set_title("Counter-assay triage: potency vs. selectivity", fontsize=14)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.15)
    return ax


def plot_single_concentration_grid(
    train_single_df: pd.DataFrame,
    *,
    concentration_col: str = "concentration_M",
    log2fc_col: str = "log2_fc_estimate",
    fdr_col: str = "fdr_bh",
    hit_log2_threshold: float = 1.0,
    hit_fdr_threshold: float = 0.05,
    bins: int = 50,
    figsize: tuple[float, float] = (12.0, 8.0),
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """
    2×2 histograms of ``log2_fc_estimate`` by concentration (primary screen; tutorial-style).

    If ``is_hit`` is absent, it is computed as
    ``(log2fc > hit_log2_threshold) & (fdr < hit_fdr_threshold)`` when ``fdr_col`` exists,
    else ``log2fc > hit_log2_threshold`` only.
    """
    required = {concentration_col, log2fc_col}
    missing = required - set(train_single_df.columns)
    if missing:
        raise ValueError(f"train_single_df missing columns: {sorted(missing)}")

    df = train_single_df.copy()
    if "is_hit" not in df.columns:
        if fdr_col in df.columns:
            df["is_hit"] = (df[log2fc_col] > hit_log2_threshold) & (df[fdr_col] < hit_fdr_threshold)
        else:
            df["is_hit"] = df[log2fc_col] > hit_log2_threshold

    concs = sorted(df[concentration_col].unique())[:4]
    if len(concs) == 0:
        raise ValueError("No concentrations in train_single_df.")

    if axes is None:
        _, axes_arr = plt.subplots(2, 2, figsize=figsize, sharex=True)
        axes_flat = axes_arr.flatten()
    else:
        axes_flat = axes.flatten()

    for index, conc in enumerate(concs):
        if index >= 4:
            break
        df_conc = df[df[concentration_col] == conc].copy()
        hit_rate = 100.0 * float(df_conc["is_hit"].mean()) if len(df_conc) else 0.0
        ax_i = axes_flat[index]
        sns.histplot(
            df_conc[log2fc_col],
            bins=bins,
            kde=True,
            ax=ax_i,
            color="steelblue",
            alpha=0.75,
        )
        ax_i.axvline(hit_log2_threshold, color="crimson", linestyle="--", label="Hit threshold")
        ax_i.set_title(f"{conc:.2g} M (N = {len(df_conc)}, hits = {hit_rate:.1f}%)")
        ax_i.set_xlabel(r"$\log_{2}$ fold change")
        ax_i.set_ylabel("Count")
        if index == 0:
            ax_i.legend()

    for j in range(len(concs), 4):
        axes_flat[j].set_visible(False)

    plt.suptitle("Primary screen: log2 fold-change by concentration", fontsize=15)
    plt.tight_layout()
    return axes_flat
