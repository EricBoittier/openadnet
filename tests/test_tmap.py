"""Integration test: kinase inhibitors -> Morgan fingerprints -> TMAP tree."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tmap.tmap import LSHForest, LayoutResult, Minhash, layout_from_lsh_forest

# -- parameters -------------------------------------------------------------
N_MOLECULES = 500  # subsample for speed
MINHASH_D = 128
FP_SIZE = 1024
MORGAN_RADIUS = 2
KNN_K = 10
KNN_KC = 10

# -- load data ---------------------------------------------------------------
data_path = Path(__file__).parent.parent / "src" / "data" / "kinase_inhibitors_after_2022.csv"
with open(data_path, newline="") as f:
    rows = list(csv.DictReader(f))
seen = set()
smiles = []
for row in rows:
    smi = row["smiles"]
    if smi not in seen:
        seen.add(smi)
        smiles.append(smi)
    if len(smiles) >= N_MOLECULES:
        break

print(f"Loaded {len(smiles)} unique molecules")

# -- Morgan fingerprints ----------------------------------------------------
gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=FP_SIZE)
fps = []
valid_idx = []
for i, smi in enumerate(smiles):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    fp = gen.GetFingerprint(mol)
    fps.append(np.array(fp, dtype=np.uint8))
    valid_idx.append(i)

fps_array = np.array(fps)
print(f"Computed fingerprints: {fps_array.shape}")

# -- MinHash -----------------------------------------------------------------
mh = Minhash(d=MINHASH_D, seed=42)
minhashes = mh.batch_from_binary_array(fps_array)
print(f"MinHashes: {minhashes.shape}")

# -- LSH Forest --------------------------------------------------------------
lf = LSHForest(d=MINHASH_D, l=8, store=True)
lf.batch_add(minhashes)
lf.index()
print(f"LSH Forest indexed: {lf.size} entries")

# -- Layout -------------------------------------------------------------------
result: LayoutResult = layout_from_lsh_forest(lf, k=KNN_K, kc=KNN_KC, fme_iterations=200)
print(f"Layout complete: MST weight={result.mst_weight:.4f}, edges={len(result.s)}")

# -- Plot ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

for si, ti in zip(result.s, result.t):
    ax.plot(
        [result.x[si], result.x[ti]],
        [result.y[si], result.y[ti]],
        color="lightgray",
        linewidth=0.4,
        zorder=1,
    )

ax.scatter(result.x, result.y, s=4, c="steelblue", zorder=2)
ax.set_title(f"TMAP – {len(smiles)} kinase inhibitors")
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout()
plt.savefig(Path(__file__).parent / "tmap_kinase.png", dpi=150)
print("Saved tmap_kinase.png")
plt.show()
