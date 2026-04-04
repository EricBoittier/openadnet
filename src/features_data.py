from __future__ import annotations

import numpy as np
from rdkit.Chem import Descriptors, rdFingerprintGenerator

FP_SIZE = 2048

# Fixed-size generators: GetFingerprintAsNumPy(mol) -> (FP_SIZE,) uint8
FP_GENERATORS: dict[str, object] = {
    "morgan_bits_2048_r2": rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=FP_SIZE
    ),
    "rdkit_bits_2048": rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=FP_SIZE),
    "atom_pair_bits_2048": rdFingerprintGenerator.GetAtomPairGenerator(
        fpSize=FP_SIZE
    ),
    "torsion_bits_2048": rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        fpSize=FP_SIZE
    ),
}

# Scalar physicochemical descriptors (per molecule)
_RDKIT_PHYS_FUNCS: list[tuple[str, object]] = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("LabuteASA", Descriptors.LabuteASA),
    ("PEOE_VSA1", Descriptors.PEOE_VSA1),
    ("BalabanJ", Descriptors.BalabanJ),
]

RDKIT_PHYS_PROP_NAMES: tuple[str, ...] = tuple(n for n, _ in _RDKIT_PHYS_FUNCS)


def _fp_zeros() -> np.ndarray:
    return np.zeros(FP_SIZE, dtype=np.uint8)


def build_fingerprint_matrix(name: str, mols: list) -> np.ndarray:
    if name not in FP_GENERATORS:
        raise KeyError(f"Unknown fingerprint descriptor: {name!r}")
    gen = FP_GENERATORS[name]
    rows = []
    for m in mols:
        if m is None:
            rows.append(_fp_zeros())
        else:
            rows.append(np.asarray(gen.GetFingerprintAsNumPy(m), dtype=np.uint8))
    return np.stack(rows, axis=0)


def build_rdkit_phys_props_matrix(mols: list) -> np.ndarray:
    n = len(_RDKIT_PHYS_FUNCS)
    out = np.empty((len(mols), n), dtype=np.float64)
    for i, m in enumerate(mols):
        if m is None:
            out[i, :] = np.nan
            continue
        for j, (_, fn) in enumerate(_RDKIT_PHYS_FUNCS):
            try:
                out[i, j] = float(fn(m))
            except Exception:
                out[i, j] = np.nan
    return out


def list_descriptor_names() -> list[str]:
    return list(FP_GENERATORS.keys()) + ["rdkit_phys_props"]


def build_descriptor_matrix(name: str, mols: list) -> np.ndarray:
    if name == "rdkit_phys_props":
        return build_rdkit_phys_props_matrix(mols)
    return build_fingerprint_matrix(name, mols)


# Backwards compatibility (notebook / older code key names)
fpspecs = {
    "morgan_bits_2048_r2": FP_GENERATORS["morgan_bits_2048_r2"],
    "rdkit_bits_2048": FP_GENERATORS["rdkit_bits_2048"],
    "atom_pair_sparse": FP_GENERATORS["atom_pair_bits_2048"],
    "torsion_sparse": FP_GENERATORS["torsion_bits_2048"],
}
