from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from joblib import Memory
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

FP_SIZE = 2048

# Bump when FP generators, FP_SIZE, or descriptor definitions change (invalidates disk cache).
FP_CACHE_VERSION = "v1_fp2048_phys15"

# Disk cache for per-molecule fingerprints / phys vectors. Set OPENADNET_FP_CACHE="" to disable
# disk (in-process lru_cache only). Default: ~/.cache/openadnet/fingerprints


def _fp_cache_dir() -> Path | None:
    raw = os.environ.get("OPENADNET_FP_CACHE")
    if raw == "":
        return None
    base = raw or str(Path.home() / ".cache" / "openadnet" / "fingerprints")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


_fp_memory: Memory | None
_cd = _fp_cache_dir()
if _cd is not None:
    _fp_memory = Memory(location=str(_cd), verbose=0)
else:
    _fp_memory = None

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


def _mol_to_canonical_smiles(mol) -> str | None:
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _fp_row_compute(cache_version: str, fp_name: str, canon_smiles: str) -> np.ndarray:
    _ = cache_version
    gen = FP_GENERATORS[fp_name]
    m = Chem.MolFromSmiles(canon_smiles)
    if m is None:
        return _fp_zeros()
    return np.asarray(gen.GetFingerprintAsNumPy(m), dtype=np.uint8)


_fp_disk = _fp_memory.cache(_fp_row_compute) if _fp_memory else _fp_row_compute


@lru_cache(maxsize=65_536)
def _fp_row(fp_name: str, canon_smiles: str) -> np.ndarray:
    return _fp_disk(FP_CACHE_VERSION, fp_name, canon_smiles)


def _phys_row_compute(cache_version: str, canon_smiles: str) -> np.ndarray:
    _ = cache_version
    n = len(_RDKIT_PHYS_FUNCS)
    out = np.empty(n, dtype=np.float64)
    m = Chem.MolFromSmiles(canon_smiles)
    if m is None:
        out[:] = np.nan
        return out
    for j, (_, fn) in enumerate(_RDKIT_PHYS_FUNCS):
        try:
            out[j] = float(fn(m))
        except Exception:
            out[j] = np.nan
    return out


_phys_disk = _phys_row_compute
if _fp_memory:
    _phys_disk = _fp_memory.cache(_phys_row_compute)


@lru_cache(maxsize=65_536)
def _phys_row(canon_smiles: str) -> np.ndarray:
    return _phys_disk(FP_CACHE_VERSION, canon_smiles)


def clear_fingerprint_caches() -> None:
    """Clear in-process LRU caches and joblib memory store (disk files remain until overwritten)."""
    _fp_row.cache_clear()
    _phys_row.cache_clear()
    if _fp_memory is not None:
        _fp_memory.clear(warn=False)


def build_fingerprint_matrix(name: str, mols: list) -> np.ndarray:
    if name not in FP_GENERATORS:
        raise KeyError(f"Unknown fingerprint descriptor: {name!r}")
    rows = []
    for m in mols:
        smi = _mol_to_canonical_smiles(m)
        if smi is None:
            rows.append(_fp_zeros())
        else:
            rows.append(_fp_row(name, smi))
    return np.stack(rows, axis=0)


def build_rdkit_phys_props_matrix(mols: list) -> np.ndarray:
    n = len(_RDKIT_PHYS_FUNCS)
    out = np.empty((len(mols), n), dtype=np.float64)
    for i, m in enumerate(mols):
        smi = _mol_to_canonical_smiles(m)
        if smi is None:
            out[i, :] = np.nan
        else:
            out[i, :] = _phys_row(smi)
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
