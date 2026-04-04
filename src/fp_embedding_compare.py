"""
Comparable 2D PCA, UMAP, and t-SNE scatter plots from a fingerprint matrix.

Uses the same viridis scale and percentile range as the TMAP activity coloring when
``values`` is provided; otherwise uses a uniform steel-blue scatter.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError as _e:  # pragma: no cover
    umap = None  # type: ignore[misc, assignment]
    _UMAP_IMPORT_ERROR = _e
else:
    _UMAP_IMPORT_ERROR = None


@dataclass(frozen=True)
class EmbeddingLayouts:
    pca: np.ndarray
    umap: np.ndarray
    tsne: np.ndarray | None


def _activity_vmin_vmax(values: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0, 1.0
    vmin = float(np.nanpercentile(values[finite], 2))
    vmax = float(np.nanpercentile(values[finite], 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def compute_2d_embeddings(
    fps: np.ndarray,
    *,
    random_state: int = 42,
    tsne_max_points: int = 5000,
) -> tuple[EmbeddingLayouts, PCA]:
    """Standard-scale fingerprints, then PCA / UMAP / (optional) t-SNE."""
    if umap is None:
        raise RuntimeError(
            "umap-learn is required for embedding comparison. "
            "Install with: pip install umap-learn"
        ) from _UMAP_IMPORT_ERROR

    n = fps.shape[0]
    X = StandardScaler().fit_transform(fps.astype(np.float64))

    pca_model = PCA(n_components=2, random_state=random_state)
    xy_pca = pca_model.fit_transform(X)

    n_neighbors = max(2, min(15, n - 1))
    umap_model = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
    )
    xy_umap = umap_model.fit_transform(X)

    tsne_xy: np.ndarray | None = None
    if n <= tsne_max_points:
        perplexity = min(30, max(5, n // 4))
        perplexity = min(perplexity, n - 1)
        tsne_model = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=float(perplexity),
            init="pca",
            learning_rate="auto",
        )
        tsne_xy = tsne_model.fit_transform(X)

    return (
        EmbeddingLayouts(pca=xy_pca, umap=xy_umap, tsne=tsne_xy),
        pca_model,
    )


def _scatter_panel(
    ax,
    xy: np.ndarray,
    values: np.ndarray | None,
    vmin: float,
    vmax: float,
    *,
    color_scale: bool,
    title: str,
    subtitle: str,
    n_total: int,
) -> None:
    s = max(2, min(18, int(8000 / max(n_total, 1))))
    if color_scale and values is not None:
        finite = np.isfinite(values)
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        ax.scatter(
            xy[finite, 0],
            xy[finite, 1],
            c=values[finite],
            cmap="viridis",
            norm=norm,
            s=s,
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )
        nan = ~finite
        if np.any(nan):
            ax.scatter(
                xy[nan, 0],
                xy[nan, 1],
                c="#888888",
                s=s,
                alpha=0.65,
                linewidths=0,
                rasterized=True,
            )
    else:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c="#4682b4",
            s=s,
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, color="#333")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("#fafafa")


def build_embedding_comparison_html(
    fps: np.ndarray,
    values: np.ndarray | None,
    *,
    value_label: str = "activity",
    random_state: int = 42,
    tsne_max_points: int = 5000,
) -> str:
    """
    Fit PCA, UMAP, and t-SNE (if ``n <= tsne_max_points``) on standardized fingerprints
    and return an HTML fragment with one matplotlib SVG figure (three panels + shared colorbar).
    """
    layouts, pca_model = compute_2d_embeddings(
        fps,
        random_state=random_state,
        tsne_max_points=tsne_max_points,
    )
    n = fps.shape[0]
    vals_arr: np.ndarray | None = None
    color_scale = False
    vmin, vmax = 0.0, 1.0
    if values is not None:
        vals_arr = np.asarray(values, dtype=float).reshape(-1)
        if len(vals_arr) != n:
            raise ValueError("values length must match fps rows")
        if np.any(np.isfinite(vals_arr)):
            color_scale = True
            vmin, vmax = _activity_vmin_vmax(vals_arr)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    ax0, ax1, ax2 = axes[0], axes[1], axes[2]

    ev = pca_model.explained_variance_ratio_
    pca_sub = f"var {100 * float(ev.sum()):.1f}%"

    _scatter_panel(
        ax0,
        layouts.pca,
        vals_arr,
        vmin,
        vmax,
        color_scale=color_scale,
        title="PCA",
        subtitle=pca_sub,
        n_total=n,
    )
    _scatter_panel(
        ax1,
        layouts.umap,
        vals_arr,
        vmin,
        vmax,
        color_scale=color_scale,
        title="UMAP",
        subtitle="",
        n_total=n,
    )

    if layouts.tsne is not None:
        _scatter_panel(
            ax2,
            layouts.tsne,
            vals_arr,
            vmin,
            vmax,
            color_scale=color_scale,
            title="t-SNE",
            subtitle="",
            n_total=n,
        )
    else:
        ax2.text(
            0.5,
            0.5,
            f"t-SNE omitted\n(n > {tsne_max_points})\n"
            "Lower --max-molecules or\n--embedding-tsne-max",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=10,
            color="#555",
        )
        ax2.set_axis_off()

    if color_scale:
        sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
        sm.set_array([])
        fig.colorbar(
            sm,
            ax=axes,
            orientation="horizontal",
            fraction=0.12,
            pad=0.08,
            label=value_label,
            shrink=0.85,
        )

    fig.patch.set_facecolor("white")

    buf = io.StringIO()
    fig.savefig(
        buf,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    svg_raw = buf.getvalue()
    svg_body = re.sub(r"<\?xml[^?]*\?>\s*", "", svg_raw, count=1).strip()

    cap = (
        "Standardized Morgan fingerprints projected with PCA, UMAP, and t-SNE where feasible. "
        "Point size, colormap, and activity range match the TMAP view above."
    )
    return (
        '<div class="tmap-embed-compare">\n'
        '<h2>Fingerprint embedding comparison</h2>\n'
        f'<p class="tmap-embed-caption">{cap}</p>\n'
        f'<div class="tmap-embed-svg-wrap">{svg_body}</div>\n'
        "</div>\n"
    )
