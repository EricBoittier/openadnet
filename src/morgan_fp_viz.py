"""
Morgan fingerprint visualization using RDKit MorganGenerator bitInfo (matches features_data).

Full-molecule highlight pattern:
https://www.andersle.no/posts/2022/drawfingerprint/drawfingerprint.html
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdFingerprintGenerator

if TYPE_CHECKING:
    from sklearn.decomposition import PCA


def _ipython_display(obj: object) -> None:
    try:
        from IPython.display import display
    except ImportError:
        if hasattr(obj, "show"):
            obj.show()
        else:
            print(obj)
    else:
        display(obj)


def morgan_bit_info(mol: Chem.Mol, radius: int, nbits: int) -> dict[int, Any]:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetFingerprint(mol, additionalOutput=ao)
    return ao.GetBitInfoMap()


def parse_morgan_params(fp_name: str) -> tuple[int, int] | None:
    m = re.match(r"morgan_r(\d+)_(bits|count)_(\d+)$", fp_name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(3))


# Okabe–Ito palette (see blog post above)
_COLOR_FRAC = [
    tuple(x / 255 for x in t)
    for t in [
        (230, 159, 0),
        (86, 180, 233),
        (0, 158, 115),
        (240, 228, 66),
        (0, 114, 178),
        (213, 94, 0),
        (204, 121, 167),
        (204, 204, 204),
    ]
]
_COLOR_MAP = {
    "Center atom": _COLOR_FRAC[1],
    "Atom in a ring": _COLOR_FRAC[6],
    "Aromatic atom": _COLOR_FRAC[3],
    "Other atoms": _COLOR_FRAC[7],
    "Bonds": _COLOR_FRAC[7],
}


def _atom_colors(mol: Chem.Mol, atoms: Sequence[int], centers: set[int] | None = None) -> dict:
    out: dict[int, tuple[float, float, float]] = {}
    for aix in atoms:
        if centers is not None and aix in centers:
            out[aix] = _COLOR_MAP["Center atom"]
        else:
            a = mol.GetAtomWithIdx(aix)
            if a.GetIsAromatic():
                out[aix] = _COLOR_MAP["Aromatic atom"]
            elif a.IsInRing():
                out[aix] = _COLOR_MAP["Atom in a ring"]
            else:
                out[aix] = _COLOR_MAP["Other atoms"]
    return out


def _bond_colors(bonds: Sequence[int]) -> dict[int, tuple[float, float, float]]:
    return {b: _COLOR_MAP["Bonds"] for b in bonds}


def _environment(mol: Chem.Mol, center: int, radius: int):
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center)
    atoms = {center}
    bond_set: set[int] = set()
    for bix in env:
        b = mol.GetBondWithIdx(bix)
        atoms.add(b.GetBeginAtomIdx())
        atoms.add(b.GetEndAtomIdx())
        bond_set.add(bix)
    al = list(atoms)
    bl = list(bond_set)
    return (
        al,
        bl,
        _atom_colors(mol, al, centers={center}),
        _bond_colors(bl),
    )


def draw_full_mol_morgan_bit(
    mol: Chem.Mol,
    info: dict,
    bit_id: int,
    *,
    max_examples: int = 2,
    sub: tuple[int, int] = (220, 220),
):
    """Full-molecule highlight for one Morgan bit (pattern from andersle.no)."""
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    hl_atoms, hl_bonds, ac, bc = [], [], [], []
    mols = []
    legends = []
    for k, ex in enumerate(info[bit_id]):
        if k >= max_examples:
            break
        c, r = ex
        atoms, bonds, atom_colors, bond_colors = _environment(m, c, r)
        hl_atoms.append(atoms)
        hl_bonds.append(bonds)
        ac.append(atom_colors)
        bc.append(bond_colors)
        mols.append(m)
        legends.append(f"bit {bit_id}  ex{k + 1}")
    opts = Draw.rdMolDraw2D.MolDrawOptions()
    opts.prepareMolsBeforeDrawing = True
    opts.fillHighlights = True
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=min(3, max(len(mols), 1)),
        subImgSize=sub,
        legends=legends,
        highlightAtomLists=hl_atoms,
        highlightBondLists=hl_bonds,
        highlightAtomColors=ac,
        highlightBondColors=bc,
        drawOptions=opts,
    )


def _best_molecule_for_bits(
    X_fp: np.ndarray,
    candidate_bits: Sequence[int],
) -> int:
    """Index of row that maximizes overlap with candidate fingerprint bits."""
    cand_set = set(int(b) for b in candidate_bits)
    best_i, best_n = 0, -1
    for i in range(X_fp.shape[0]):
        row_on = set(np.flatnonzero(X_fp[i] > 0))
        n_hit = len(row_on & cand_set)
        if n_hit > best_n:
            best_n, best_i = n_hit, i
    return best_i


def display_morgan_examples_for_impactful_pcs(
    *,
    pca_fp: PCA,
    X_fp: np.ndarray,
    mols_f: list,
    fp_name: str,
    train_smiles: Sequence[str],
    pc_permutation_importance: np.ndarray,
    n_top_pcs: int = 5,
    n_bits_ranking: int = 16,
    n_bits_morgan_grid: int = 16,
    n_bits_full_mol: int = 4,
) -> None:
    """
    For the *n_top_pcs* PCs with highest permutation importance (from RF on PCA scores),
    show Morgan fragment grids and full-molecule highlights using bits with largest |loading|
    on each of those PCs.

    Pair this with the PCA / permutation-importance bar plots for fingerprint PCs.
    """
    mp = parse_morgan_params(fp_name)
    if mp is None:
        print(
            "Skip structure drawing: fp_name is not morgan_r*_{bits|count}_* — "
            "use e.g. morgan_r2_bits_2048 (MorganGenerator settings as in features_data)."
        )
        return

    morgan_radius, morgan_nbits = mp
    evr = pca_fp.explained_variance_ratio_
    n_pc_avail = min(len(pc_permutation_importance), pca_fp.components_.shape[0])
    imp = np.asarray(pc_permutation_importance)[:n_pc_avail]
    order = np.argsort(imp)[::-1][:n_top_pcs]

    for rank, pc_i in enumerate(order):
        loadings = pca_fp.components_[pc_i]
        top_bits = np.argsort(np.abs(loadings))[-n_bits_ranking:][::-1].astype(int)
        candidate_bits = [int(b) for b in top_bits]

        best_i = _best_molecule_for_bits(X_fp, candidate_bits)
        ref = Chem.Mol(mols_f[best_i])
        info = morgan_bit_info(ref, morgan_radius, morgan_nbits)
        present = [b for b in candidate_bits if b in info]

        ev = float(evr[pc_i]) if pc_i < len(evr) else float("nan")
        imp_val = float(imp[pc_i]) if pc_i < len(imp) else float("nan")
        smi = str(train_smiles[best_i])
        smi_short = smi if len(smi) <= 100 else smi[:100] + "…"

        print(
            f"\n--- PC{pc_i + 1} (rank {rank + 1}/{n_top_pcs} by permutation ΔR²) — "
            f"explained var={ev:.3f}, perm_imp={imp_val:.5f} ---"
        )
        print(
            f"molecule idx {best_i} hits {len(present)}/{len(candidate_bits)} |loading| bits — {smi_short}"
        )

        if not present:
            print("No on-bits in bitInfo for this molecule; try another compound or fewer bits.")
            continue

        grid = present[:n_bits_morgan_grid]
        on_bits = [(ref, b, info) for b in grid]
        _ipython_display(
            Draw.DrawMorganBits(
                on_bits,
                molsPerRow=4,
                legends=[str(b[1]) for b in on_bits],
            )
        )
        for b in grid[:n_bits_full_mol]:
            _ipython_display(draw_full_mol_morgan_bit(ref, info, b, max_examples=2))
