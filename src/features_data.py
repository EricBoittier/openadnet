from __future__ import annotations

import os
from typing import Sequence, Union
from functools import lru_cache
from pathlib import Path

import numpy as np
from joblib import Memory
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# Bump when generator definitions or phys props change (invalidates joblib fingerprint cache).
FP_CACHE_VERSION = "v3_morgan_radii_0123"

# Default fingerprint widths to evaluate (smaller → larger).
DEFAULT_FP_SIZES: tuple[int, ...] = (512, 1024, 2048)

# Morgan fingerprint radii (hop / bond-radius in RDKit Morgan implementation).
DEFAULT_MORGAN_RADII: tuple[int, ...] = (0, 1, 2)

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


def _build_fp_registry(
    fp_sizes: tuple[int, ...] = DEFAULT_FP_SIZES,
    morgan_radii: tuple[int, ...] = DEFAULT_MORGAN_RADII,
) -> dict[str, dict]:
    """
    Registry keys: Morgan ``morgan_r{radius}_{bits|count}_{fp_size}``; others
    ``{family}_{bits|count}_{fp_size}``.
    Each value: gen, count flag, fp_size (for zero vectors).
    """
    reg: dict[str, dict] = {}
    for fp_size in fp_sizes:
        for radius in morgan_radii:
            gm = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=fp_size
            )
            reg[f"morgan_r{radius}_bits_{fp_size}"] = {
                "gen": gm,
                "count": False,
                "fp_size": fp_size,
            }
            reg[f"morgan_r{radius}_count_{fp_size}"] = {
                "gen": gm,
                "count": True,
                "fp_size": fp_size,
            }

        gr = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)
        reg[f"rdkit_bits_{fp_size}"] = {"gen": gr, "count": False, "fp_size": fp_size}
        reg[f"rdkit_count_{fp_size}"] = {"gen": gr, "count": True, "fp_size": fp_size}

        ga = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fp_size)
        reg[f"atom_pair_bits_{fp_size}"] = {
            "gen": ga,
            "count": False,
            "fp_size": fp_size,
        }
        reg[f"atom_pair_count_{fp_size}"] = {
            "gen": ga,
            "count": True,
            "fp_size": fp_size,
        }

        gt = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fp_size)
        reg[f"torsion_bits_{fp_size}"] = {
            "gen": gt,
            "count": False,
            "fp_size": fp_size,
        }
        reg[f"torsion_count_{fp_size}"] = {
            "gen": gt,
            "count": True,
            "fp_size": fp_size,
        }

    return reg


FP_REGISTRY: dict[str, dict] = _build_fp_registry()

# Legacy / notebook names → canonical registry keys
_NAME_ALIASES: dict[str, str] = {
    "morgan_bits_2048_r2": "morgan_r2_bits_2048",
    "rdkit_bits_2048": "rdkit_bits_2048",
    "atom_pair_bits_2048": "atom_pair_bits_2048",
    "atom_pair_sparse": "atom_pair_bits_2048",
    "torsion_bits_2048": "torsion_bits_2048",
    "torsion_sparse": "torsion_bits_2048",
}


def resolve_descriptor_name(name: str) -> str:
    return _NAME_ALIASES.get(name, name)


# Backwards compatibility: fpspecs maps old keys to generators (2048 bits only)
def _fpspecs_compat() -> dict[str, object]:
    return {
        "morgan_bits_2048_r2": FP_REGISTRY["morgan_r2_bits_2048"]["gen"],
        "rdkit_bits_2048": FP_REGISTRY["rdkit_bits_2048"]["gen"],
        "atom_pair_sparse": FP_REGISTRY["atom_pair_bits_2048"]["gen"],
        "torsion_sparse": FP_REGISTRY["torsion_bits_2048"]["gen"],
    }


fpspecs = _fpspecs_compat()

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


def _fp_zeros(fp_size: int, count: bool) -> np.ndarray:
    if count:
        return np.zeros(fp_size, dtype=np.float32)
    return np.zeros(fp_size, dtype=np.uint8)


def _mol_to_canonical_smiles(mol) -> str | None:
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _fp_row_compute(cache_version: str, fp_name: str, canon_smiles: str) -> np.ndarray:
    _ = cache_version
    canon = resolve_descriptor_name(fp_name)
    if canon not in FP_REGISTRY:
        raise KeyError(f"Unknown fingerprint descriptor: {fp_name!r}")
    entry = FP_REGISTRY[canon]
    gen = entry["gen"]
    count = entry["count"]
    fp_size = entry["fp_size"]
    m = Chem.MolFromSmiles(canon_smiles)
    if m is None:
        return _fp_zeros(fp_size, count)
    if count:
        arr = np.asarray(gen.GetCountFingerprintAsNumPy(m), dtype=np.float32)
        return arr
    return np.asarray(gen.GetFingerprintAsNumPy(m), dtype=np.uint8)


_fp_disk = _fp_memory.cache(_fp_row_compute) if _fp_memory else _fp_row_compute


@lru_cache(maxsize=131_072)
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
    canon = resolve_descriptor_name(name)
    if canon not in FP_REGISTRY:
        raise KeyError(f"Unknown fingerprint descriptor: {name!r}")
    entry = FP_REGISTRY[canon]
    fp_size = entry["fp_size"]
    count = entry["count"]
    rows = []
    for m in mols:
        smi = _mol_to_canonical_smiles(m)
        if smi is None:
            rows.append(_fp_zeros(fp_size, count))
        else:
            rows.append(_fp_row(canon, smi))
    stacked = np.stack(rows, axis=0)
    if count:
        return stacked.astype(np.float64, copy=False)
    return stacked.astype(np.float64, copy=False)


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


def list_descriptor_names(
    fp_sizes: tuple[int, ...] | None = None,
    morgan_radii: tuple[int, ...] | None = None,
    include_phys: bool = True,
) -> list[str]:
    """
    All fingerprint descriptor keys for the given sizes (default: :data:`DEFAULT_FP_SIZES`)
    and Morgan radii (default: :data:`DEFAULT_MORGAN_RADII`).
    """
    sizes = fp_sizes if fp_sizes is not None else DEFAULT_FP_SIZES
    radii = morgan_radii if morgan_radii is not None else DEFAULT_MORGAN_RADII
    keys: list[str] = []
    for fp_size in sizes:
        for radius in radii:
            keys.extend(
                [
                    f"morgan_r{radius}_bits_{fp_size}",
                    f"morgan_r{radius}_count_{fp_size}",
                ]
            )
        keys.extend(
            [
                f"rdkit_bits_{fp_size}",
                f"rdkit_count_{fp_size}",
                f"atom_pair_bits_{fp_size}",
                f"atom_pair_count_{fp_size}",
                f"torsion_bits_{fp_size}",
                f"torsion_count_{fp_size}",
            ]
        )
    if include_phys:
        keys.append("rdkit_phys_props")
    return keys


def build_descriptor_matrix(name: str, mols: list) -> np.ndarray:
    if name == "rdkit_phys_props":
        return build_rdkit_phys_props_matrix(mols)
    return build_fingerprint_matrix(name, mols)


def descriptor_dim(name: str) -> int:
    """Number of columns produced by :func:`build_descriptor_matrix` for ``name``."""
    canon = resolve_descriptor_name(name)
    if canon == "rdkit_phys_props":
        return len(_RDKIT_PHYS_FUNCS)
    if canon not in FP_REGISTRY:
        raise KeyError(f"Unknown descriptor: {name!r}")
    return int(FP_REGISTRY[canon]["fp_size"])


def descriptor_dim_total(names: Sequence[str]) -> int:
    """Sum of :func:`descriptor_dim` for each name (concatenated descriptor width)."""
    return sum(descriptor_dim(n) for n in names)


def descriptor_dim_for_names(names: Union[str, Sequence[str]]) -> int:
    """Width of one descriptor, or concatenated width when ``names`` is a list/tuple (order preserved)."""
    if isinstance(names, str):
        return descriptor_dim(names)
    return descriptor_dim_total(names)


# Legacy name for single constant (used in tests / imports)
FP_SIZE = 2048
FP_GENERATORS = {k: v["gen"] for k, v in FP_REGISTRY.items() if "bits" in k}
