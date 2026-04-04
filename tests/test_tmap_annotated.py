"""TMAP tree annotated with molecule images — direct SVG output, no matplotlib."""

import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

t0 = time.time()


def tick(msg: str) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


tick("Importing numpy ...")
import numpy as np
tick("Importing rdkit ...")
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
tick("Importing tmap ...")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tmap.tmap import LSHForest, Minhash, layout_from_lsh_forest
tick("All imports done")

# -- parameters -------------------------------------------------------------
N_MOLECULES = 39847
MINHASH_D = 128
FP_SIZE = 2048
MORGAN_RADIUS = 3
KNN_K = 10
KNN_KC = 5
SVG_SIZE = 8000  # canvas pixels
REPEL_ITERS = 200
SHOW_VORONOI = False  # draw Voronoi cell polygons
SHOW_MOL_STRUCTURES = False  # draw individual molecule SVGs (singletons)
MIN_SCAFFOLD_GROUP = 20  # only show scaffold card for groups with >= N members
SCAFFOLD_TYPE = "murcko"  # "murcko", "generic", or "brics"

# tab20 palette (hex)
TAB20 = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
]

# -- load data ---------------------------------------------------------------
tick("Loading CSV ...")
data_path = (
    Path(__file__).parent.parent / "src" / "data" / "kinase_inhibitors_after_2022.csv"
)
with open(data_path, newline="") as f:
    rows = list(csv.DictReader(f))
seen: set[str] = set()
smiles_list: list[str] = []
for row in rows:
    smi = row["smiles"]
    if smi not in seen:
        seen.add(smi)
        smiles_list.append(smi)
    if len(smiles_list) >= N_MOLECULES:
        break
tick(f"Loaded {len(smiles_list)} unique molecules")

# -- fingerprints & valid mols -----------------------------------------------
tick("Computing Morgan fingerprints ...")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=FP_SIZE)
fps, mols = [], []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    fps.append(np.array(gen.GetFingerprint(mol), dtype=np.uint8))
    mols.append(mol)

fps_array = np.array(fps)
n = len(mols)
tick(f"Fingerprints done: {n} molecules, shape {fps_array.shape}")

# -- scaffold decomposition --------------------------------------------------
tick(f"Computing scaffolds (type={SCAFFOLD_TYPE}) ...")
scaffold_smi_per_mol: list[str] = []
scaffold_mol_map: dict[str, Chem.Mol] = {}

if SCAFFOLD_TYPE == "brics":
    from rdkit.Chem import BRICS

for mol in mols:
    try:
        if SCAFFOLD_TYPE == "murcko":
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            smi = Chem.MolToSmiles(scaf)
            if smi and smi not in scaffold_mol_map:
                scaffold_mol_map[smi] = scaf
        elif SCAFFOLD_TYPE == "generic":
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
            smi = Chem.MolToSmiles(scaf)
            if smi and smi not in scaffold_mol_map:
                scaffold_mol_map[smi] = scaf
        elif SCAFFOLD_TYPE == "brics":
            frags = BRICS.BRICSDecompose(mol, minFragmentSize=5)
            smi = sorted(frags, key=len, reverse=True)[0] if frags else ""
            if smi and smi not in scaffold_mol_map:
                frag_mol = Chem.MolFromSmiles(smi)
                if frag_mol:
                    scaffold_mol_map[smi] = frag_mol
        else:
            raise ValueError(f"Unknown SCAFFOLD_TYPE: {SCAFFOLD_TYPE}")
    except Exception:
        smi = ""
    scaffold_smi_per_mol.append(smi)

scaffold_groups: dict[str, list[int]] = defaultdict(list)
for i, smi in enumerate(scaffold_smi_per_mol):
    scaffold_groups[smi].append(i)

sorted_scaffolds = sorted(scaffold_groups.items(), key=lambda kv: -len(kv[1]))
scaffold_color_map: dict[str, int] = {smi: ci for ci, (smi, _) in enumerate(sorted_scaffolds)}
mol_scaffold_color = [scaffold_color_map[scaffold_smi_per_mol[i]] for i in range(n)]

tick(f"Scaffolds: {len(scaffold_mol_map)} unique from {n} molecules")

# -- TMAP pipeline -----------------------------------------------------------
tick("MinHashing ...")
mh = Minhash(d=MINHASH_D, seed=42)
minhashes = mh.batch_from_binary_array(fps_array)

tick("Building LSH Forest + index ...")
lf = LSHForest(d=MINHASH_D, l=8, store=True)
lf.batch_add(minhashes)
lf.index()

tick("Computing layout (kNN -> MST -> spring) ...")
result = layout_from_lsh_forest(lf, k=KNN_K, kc=KNN_KC, fme_iterations=150)
tick(f"Layout done: {len(result.s)} edges")

# -- adaptive cell size ------------------------------------------------------
data_range = 1.0
ideal_cell = data_range / np.sqrt(n)
target_data_size = ideal_cell * 0.92
box_half = target_data_size / 2
tick(f"Cell={ideal_cell:.4f}  box_half={box_half:.4f}")

# -- grid-accelerated force repulsion ----------------------------------------
tick(f"Repelling labels ({REPEL_ITERS} iters, grid-accelerated) ...")
origins = np.column_stack([result.x, result.y]).astype(np.float64)
positions = origins.copy()
bh2 = 2.0 * box_half
_GRID_DIRS = ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 1))

for it in range(REPEL_ITERS):
    inv_cs = 1.0 / bh2
    px = positions[:, 0]
    py = positions[:, 1]
    gx_all = np.floor(px * inv_cs).astype(np.intp).tolist()
    gy_all = np.floor(py * inv_cs).astype(np.intp).tolist()

    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(n):
        grid[(gx_all[i], gy_all[i])].append(i)

    src: list[int] = []
    dst: list[int] = []
    for (gx, gy), cell in grid.items():
        for dgx, dgy in _GRID_DIRS:
            nb = grid.get((gx + dgx, gy + dgy))
            if nb is None:
                continue
            same = dgx == 0 and dgy == 0
            for ki, i in enumerate(cell):
                for j in (cell[ki + 1 :] if same else nb):
                    src.append(i)
                    dst.append(j)

    fx = np.zeros(n)
    fy = np.zeros(n)
    n_overlaps = 0

    if src:
        sa = np.array(src, dtype=np.intp)
        da = np.array(dst, dtype=np.intp)
        ddx = px[sa] - px[da]
        ddy = py[sa] - py[da]
        ox = bh2 - np.abs(ddx)
        oy = bh2 - np.abs(ddy)
        m = (ox > 0) & (oy > 0)
        n_overlaps = int(m.sum())

        px_push = m & (ox < oy)
        py_push = m & ~px_push
        sdx = np.sign(ddx + 1e-12)
        sdy = np.sign(ddy + 1e-12)
        fxi = np.where(px_push, ox * sdx, 0.0)
        fyi = np.where(py_push, oy * sdy, 0.0)
        np.add.at(fx, sa, fxi)
        np.add.at(fx, da, -fxi)
        np.add.at(fy, sa, fyi)
        np.add.at(fy, da, -fyi)

    fx += 0.008 * (origins[:, 0] - px)
    fy += 0.008 * (origins[:, 1] - py)

    step = 0.35 if n_overlaps > n else 0.18
    positions[:, 0] += step * fx
    positions[:, 1] += step * fy

    if it % 100 == 0:
        tick(f"  iter {it:4d}  overlaps={n_overlaps}")

tick(f"Repulsion done ({REPEL_ITERS} iters)")

# -- Voronoi (KNN-limited, half-plane clipping) -----------------------------
tick("Computing Voronoi (KNN-accelerated) ...")

K_VORONOI = 50
_VCHUNK = min(300, n)

tick("  finding approximate nearest neighbours ...")
knn_vor = np.zeros((n, min(K_VORONOI, n - 1)), dtype=np.intp)
for i0 in range(0, n, _VCHUNK):
    i1 = min(i0 + _VCHUNK, n)
    cs = i1 - i0
    dx = positions[i0:i1, 0:1] - positions[:, 0:1].T
    dy = positions[i0:i1, 1:2] - positions[:, 1:2].T
    d2 = dx * dx + dy * dy
    d2[np.arange(cs), np.arange(i0, i1)] = np.inf
    k_actual = knn_vor.shape[1]
    knn_vor[i0:i1] = np.argpartition(d2, k_actual, axis=1)[:, :k_actual]
tick("  KNN done")


def _clip_polygon_by_halfplane(poly, nx_, ny_, d):
    """Clip polygon (Nx2 array) by half-plane nx*x + ny*y <= d."""
    if len(poly) == 0:
        return None
    dots = poly[:, 0] * nx_ + poly[:, 1] * ny_
    inside = dots <= d + 1e-12
    if inside.all():
        return poly
    if not inside.any():
        return None
    clipped = []
    n_pts = len(poly)
    for j in range(n_pts):
        k = (j + 1) % n_pts
        pj, pk = poly[j], poly[k]
        dj, dk = dots[j] - d, dots[k] - d
        if dj <= 1e-12:
            clipped.append(pj)
        if (dj < -1e-12) != (dk < -1e-12):
            t = dj / (dj - dk)
            clipped.append(pj + t * (pk - pj))
    return np.array(clipped) if clipped else None


CELL_RADIUS_MULT = 3.0  # max cell radius = this × median nearest-neighbor distance
_N_CIRCLE_SIDES = 32
_CIRCLE_ANGLES = np.linspace(0, 2 * np.pi, _N_CIRCLE_SIDES, endpoint=False)

# compute max cell radius from median 1-NN distance
nn1_dists = np.zeros(n)
for i0 in range(0, n, _VCHUNK):
    i1 = min(i0 + _VCHUNK, n)
    nb = knn_vor[i0:i1]
    d2 = np.sum((positions[nb] - positions[i0:i1, None, :]) ** 2, axis=2)
    nn1_dists[i0:i1] = np.sqrt(d2.min(axis=1))
max_cell_radius = float(np.median(nn1_dists) * CELL_RADIUS_MULT)
tick(f"  max cell radius = {max_cell_radius:.4f} (median NN = {np.median(nn1_dists):.4f})")


def voronoi_cell_knn(idx, points, neighbors, max_r):
    """Voronoi cell starting from a circle, clipped against K nearest neighbors."""
    pi = points[idx]
    poly = np.column_stack([
        pi[0] + max_r * np.cos(_CIRCLE_ANGLES),
        pi[1] + max_r * np.sin(_CIRCLE_ANGLES),
    ])
    nb_dists = np.sum((points[neighbors] - pi) ** 2, axis=1)
    sorted_nb = neighbors[np.argsort(nb_dists)]
    for j in sorted_nb:
        pj = points[j]
        mx, my = (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2
        nx_ = pj[0] - pi[0]
        ny_ = pj[1] - pi[1]
        d = nx_ * mx + ny_ * my
        poly = _clip_polygon_by_halfplane(poly, nx_, ny_, d)
        if poly is None:
            break
    return poly

voronoi_polys: list[np.ndarray] = []
voronoi_color_idx: list[int] = []
for i in range(n):
    cell = voronoi_cell_knn(i, positions, knn_vor[i], max_cell_radius)
    if cell is not None and len(cell) >= 3:
        voronoi_polys.append(cell)
        voronoi_color_idx.append(i)
tick(f"Voronoi cells: {len(voronoi_polys)}/{n}")

# -- merge Voronoi cells by Murcko scaffold (convex hull) -------------------
tick("Merging Voronoi cells by scaffold ...")


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. Returns CCW hull vertices as Nx2 array."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return np.array(pts)

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


voronoi_lookup: dict[int, np.ndarray] = {}
for poly, ci in zip(voronoi_polys, voronoi_color_idx):
    voronoi_lookup[ci] = poly

merged_hulls: dict[str, np.ndarray] = {}
merged_centroids: dict[str, np.ndarray] = {}
for smi, members in scaffold_groups.items():
    if len(members) < MIN_SCAFFOLD_GROUP:
        continue
    all_verts = [voronoi_lookup[idx] for idx in members if idx in voronoi_lookup]
    if not all_verts:
        continue
    all_verts_arr = np.vstack(all_verts)
    hull = _convex_hull_2d(all_verts_arr)
    if len(hull) >= 3:
        merged_hulls[smi] = hull
        merged_centroids[smi] = hull.mean(axis=0)

tick(f"Merged scaffold regions: {len(merged_hulls)} (groups >= {MIN_SCAFFOLD_GROUP})")

# -- coordinate mapping to SVG pixels ---------------------------------------
pad = 0.05
all_pts = np.vstack([origins, positions])
x_min, y_min = all_pts.min(axis=0) - pad
x_max, y_max = all_pts.max(axis=0) + pad
data_w = x_max - x_min
data_h = y_max - y_min
scale = SVG_SIZE / max(data_w, data_h)
canvas_w = data_w * scale
canvas_h = data_h * scale

cell_px = target_data_size * scale


def to_svg(x: float, y: float) -> tuple[float, float]:
    return (x - x_min) * scale, (y - y_min) * scale


# -- molecule SVG strings ----------------------------------------------------
tick(f"Rendering {n} molecule SVGs ...")

_STRIP_RE = re.compile(
    r"<\?xml[^?]*\?>|<!DOCTYPE[^>]*>|<svg[^>]*>|</svg>",
    re.IGNORECASE,
)

MOL_DRAW_SIZE = 200  # internal drawing units for MolDraw2DSVG


def mol_to_svg_inner(mol) -> str:
    """Render molecule to SVG inner content (no xml decl, no outer svg tags)."""
    drawer = rdMolDraw2D.MolDraw2DSVG(MOL_DRAW_SIZE, MOL_DRAW_SIZE)
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return _STRIP_RE.sub("", svg).strip()


# only render SVGs for molecules NOT covered by a scaffold group
grouped_indices: set[int] = set()
for smi, members in scaffold_groups.items():
    if smi in merged_hulls:
        grouped_indices.update(members)

mol_svgs: dict[int, str] = {}
if SHOW_MOL_STRUCTURES:
    singleton_indices = [i for i in range(n) if i not in grouped_indices]
    for i in singleton_indices:
        mol_svgs[i] = mol_to_svg_inner(mols[i])
    tick(f"Molecule SVGs done: {len(mol_svgs)} singletons (skipped {len(grouped_indices)} grouped)")
else:
    tick("Molecule SVGs skipped (SHOW_MOL_STRUCTURES=False)")

tick(f"Rendering {len(merged_hulls)} scaffold SVGs ...")
scaffold_svgs: dict[str, str] = {}
for smi, scaf_mol in scaffold_mol_map.items():
    if smi in merged_hulls:
        scaffold_svgs[smi] = mol_to_svg_inner(scaf_mol)
tick("Scaffold SVGs done")


# -- build SVG ---------------------------------------------------------------
tick("Assembling SVG ...")
parts: list[str] = []
parts.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'viewBox="0 0 {canvas_w:.1f} {canvas_h:.1f}" '
    f'width="{canvas_w:.0f}" height="{canvas_h:.0f}">\n'
)
parts.append('<rect width="100%" height="100%" fill="white"/>\n')

# --- Voronoi cells (colored by scaffold group) ---
if SHOW_VORONOI:
    parts.append("<!-- Voronoi cells -->\n")
    for poly, ci in zip(voronoi_polys, voronoi_color_idx):
        pts_str = " ".join(f"{to_svg(x, y)[0]:.1f},{to_svg(x, y)[1]:.1f}" for x, y in poly)
        colour = TAB20[mol_scaffold_color[ci] % 20]
        parts.append(
            f'<polygon points="{pts_str}" fill="{colour}" fill-opacity="0.18" '
            f'stroke="#b3b3b3" stroke-opacity="0.3" stroke-width="0.5"/>\n'
        )

# --- merged scaffold region outlines ---
if SHOW_VORONOI:
    parts.append("<!-- Merged scaffold regions -->\n")
    for smi, hull in merged_hulls.items():
        if len(scaffold_groups[smi]) < MIN_SCAFFOLD_GROUP:
            continue
        ci = scaffold_color_map[smi]
        colour = TAB20[ci % 20]
        pts_str = " ".join(
            f"{to_svg(x, y)[0]:.1f},{to_svg(x, y)[1]:.1f}" for x, y in hull
        )
        parts.append(
            f'<polygon points="{pts_str}" fill="{colour}" fill-opacity="0.06" '
            f'stroke="{colour}" stroke-width="3" stroke-opacity="0.7" '
            f'stroke-linejoin="round"/>\n'
        )

# --- tree edges ---
parts.append("<!-- Tree edges -->\n")
for si, ti in zip(result.s, result.t):
    x1, y1 = to_svg(origins[si, 0], origins[si, 1])
    x2, y2 = to_svg(origins[ti, 0], origins[ti, 1])
    parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="#aaaaaa" stroke-width="1.2"/>\n'
    )

# --- connector lines (origin -> displaced label) ---
parts.append("<!-- Connector lines -->\n")
for i in range(n):
    dist = np.hypot(positions[i, 0] - origins[i, 0], positions[i, 1] - origins[i, 1])
    if dist > box_half * 0.3:
        x1, y1 = to_svg(origins[i, 0], origins[i, 1])
        x2, y2 = to_svg(positions[i, 0], positions[i, 1])
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#4682b4" stroke-width="0.8" stroke-opacity="0.4"/>\n'
        )

# --- node dots ---
parts.append("<!-- Node dots -->\n")
for i in range(n):
    cx, cy = to_svg(origins[i, 0], origins[i, 1])
    parts.append(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" '
        f'fill="#4682b4" fill-opacity="0.7"/>\n'
    )

# --- molecule images inscribed inside Voronoi cells ---
parts.append("<!-- Molecule structures -->\n")


def _polygon_inradius(poly: np.ndarray) -> tuple[float, float, float]:
    """Centroid and inradius (min distance from centroid to any edge) of a convex polygon."""
    cx = float(poly[:, 0].mean())
    cy = float(poly[:, 1].mean())
    n_pts = len(poly)
    min_dist = float("inf")
    for j in range(n_pts):
        k = (j + 1) % n_pts
        ex, ey = poly[k, 0] - poly[j, 0], poly[k, 1] - poly[j, 1]
        elen = np.hypot(ex, ey)
        if elen < 1e-12:
            continue
        dist = abs((cx - poly[j, 0]) * ey - (cy - poly[j, 1]) * ex) / elen
        min_dist = min(min_dist, dist)
    return cx, cy, min_dist


# precompute per-cell: centroid (SVG coords) and inscribed side
cell_fit: dict[int, tuple[float, float, float]] = {}
all_inradii: list[float] = []
for poly, ci in zip(voronoi_polys, voronoi_color_idx):
    dcx, dcy, dr = _polygon_inradius(poly)
    scx, scy = to_svg(dcx, dcy)
    s = dr * scale * 2.0 * 0.88
    cell_fit[ci] = (scx, scy, s)
    all_inradii.append(s)

_median_inr = float(np.median(all_inradii)) if all_inradii else cell_px
max_mol_side = min(_median_inr * 1.5, SVG_SIZE / np.sqrt(n) * 1.2)

if SHOW_MOL_STRUCTURES:
    for i in mol_svgs:
        if i in cell_fit:
            cx_svg, cy_svg, side = cell_fit[i]
        else:
            cx_svg, cy_svg = to_svg(positions[i, 0], positions[i, 1])
            side = cell_px * 0.88

        side = min(side, max_mol_side)

        sx = cx_svg - side / 2
        sy = cy_svg - side / 2
        parts.append(
            f'<svg x="{sx:.1f}" y="{sy:.1f}" '
            f'width="{side:.1f}" height="{side:.1f}" '
            f'viewBox="0 0 {MOL_DRAW_SIZE} {MOL_DRAW_SIZE}" '
            f'overflow="hidden">\n'
        )
        parts.append(mol_svgs[i])
        parts.append("\n</svg>\n")

# --- scaffold images in merged regions (inscribed in hull) ---
parts.append("<!-- Scaffold structures (merged regions) -->\n")
SCAFFOLD_DRAW_SIZE = 250
SCAFFOLD_REPEL_ITERS = 200

# collect scaffold cards: (smi, cx, cy, side)
scaffold_cards: list[tuple[str, float, float, float]] = []
for smi, hull in merged_hulls.items():
    if smi not in scaffold_svgs:
        continue
    if len(scaffold_groups[smi]) < MIN_SCAFFOLD_GROUP:
        continue
    hull_svg = np.array([to_svg(x, y) for x, y in hull])
    hcx = float(hull_svg[:, 0].mean())
    hcy = float(hull_svg[:, 1].mean())
    hw = float(hull_svg[:, 0].max() - hull_svg[:, 0].min())
    hh = float(hull_svg[:, 1].max() - hull_svg[:, 1].min())
    side = min(hw, hh) * 0.88
    side = max(side, 80.0)
    scaffold_cards.append((smi, hcx, hcy, side))

if scaffold_cards:
    tick(f"Repelling {len(scaffold_cards)} scaffold cards ({SCAFFOLD_REPEL_ITERS} iters) ...")
    sc_pos = np.array([[cx, cy] for _, cx, cy, _ in scaffold_cards])
    sc_origins = sc_pos.copy()
    sc_sides = np.array([s for _, _, _, s in scaffold_cards])
    sc_n = len(scaffold_cards)

    for it in range(SCAFFOLD_REPEL_ITERS):
        ddx = sc_pos[:, 0, None] - sc_pos[None, :, 0]
        ddy = sc_pos[:, 1, None] - sc_pos[None, :, 1]
        # overlap threshold: half-side_i + half-side_j
        thresh_x = (sc_sides[:, None] + sc_sides[None, :]) / 2.0
        thresh_y = thresh_x.copy()
        ox = thresh_x - np.abs(ddx)
        oy = thresh_y - np.abs(ddy)

        mask = np.triu((ox > 0) & (oy > 0), k=1)
        push_x = mask & (ox < oy)
        push_y = mask & ~push_x & mask
        sdx = np.sign(ddx + 1e-12)
        sdy = np.sign(ddy + 1e-12)

        fx = (push_x * ox * sdx).sum(axis=1) - (push_x * ox * sdx).sum(axis=0)
        fy = (push_y * oy * sdy).sum(axis=1) - (push_y * oy * sdy).sum(axis=0)

        fx += 0.01 * (sc_origins[:, 0] - sc_pos[:, 0])
        fy += 0.01 * (sc_origins[:, 1] - sc_pos[:, 1])

        n_olap = int(mask.sum())
        step = 0.4 if n_olap > sc_n else 0.2
        sc_pos[:, 0] += step * fx
        sc_pos[:, 1] += step * fy

    tick(f"Scaffold repulsion done — {int(mask.sum())} remaining overlaps")

    for idx, (smi, _, _, side) in enumerate(scaffold_cards):
        cx_s, cy_s = float(sc_pos[idx, 0]), float(sc_pos[idx, 1])
        sx = cx_s - side / 2
        sy = cy_s - side / 2

        ci = scaffold_color_map[smi]
        colour = TAB20[ci % 20]
        group_size = len(scaffold_groups[smi])

        parts.append(
            f'<svg x="{sx:.1f}" y="{sy:.1f}" '
            f'width="{side:.1f}" height="{side:.1f}" '
            f'viewBox="0 0 {SCAFFOLD_DRAW_SIZE} {SCAFFOLD_DRAW_SIZE}" '
            f'overflow="hidden">\n'
        )
        parts.append(scaffold_svgs[smi])
        parts.append("\n</svg>\n")

        parts.append(
            f'<text x="{cx_s:.1f}" y="{sy + side + 14:.1f}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="11" '
            f'fill="{colour}" font-weight="bold">n={group_size}</text>\n'
        )

# --- title ---
parts.append(
    f'<text x="{canvas_w / 2:.0f}" y="30" text-anchor="middle" '
    f'font-family="sans-serif" font-size="28" fill="#333">'
    f'TMAP \u2013 {n} kinase inhibitors</text>\n'
)

parts.append("</svg>\n")

svg_content = "".join(parts)
tick(f"SVG assembled: {len(svg_content) / 1024:.0f} KB")

# -- write SVG ---------------------------------------------------------------
out_svg = Path(__file__).parent / "tmap_kinase_annotated.svg"
out_svg.write_text(svg_content, encoding="utf-8")
tick(f"Saved {out_svg}")

# -- write HTML wrapper for easy viewing ------------------------------------
out_html = Path(__file__).parent / "tmap_kinase_annotated.html"
out_html.write_text(
    "<!DOCTYPE html>\n"
    '<html><head><meta charset="utf-8">'
    f"<title>TMAP – {n} kinase inhibitors</title>"
    "<style>body{margin:0;display:flex;justify-content:center;"
    "align-items:center;min-height:100vh;background:#fafafa}"
    "svg{max-width:100vw;max-height:100vh}</style></head><body>\n"
    + svg_content
    + "</body></html>\n",
    encoding="utf-8",
)
tick(f"Saved {out_html}")

# -- optional PDF via cairosvg -----------------------------------------------
try:
    import cairosvg
    tick("Converting SVG to PDF ...")
    out_pdf = Path(__file__).parent / "tmap_kinase_annotated.pdf"
    cairosvg.svg2pdf(bytestring=svg_content.encode("utf-8"),
                     write_to=str(out_pdf))
    tick(f"Saved {out_pdf}")
except ImportError:
    tick("cairosvg not installed — skipping PDF (install with: uv add cairosvg)")

tick("Done")
