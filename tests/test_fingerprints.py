from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

test_smiles = "CCO"

mol = Chem.MolFromSmiles(test_smiles)

print(mol)

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
fingerprint = generator.GetFingerprint(mol)

print(fingerprint)

fingerprints_func = lambda x: generator.GetFingerprint(Chem.MolFromSmiles(x))