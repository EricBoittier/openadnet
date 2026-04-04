import rdkit

from rdkit import Chem

test_smiles = "CCO"

print("RDKit version:", rdkit.__version__)
mol = Chem.MolFromSmiles(test_smiles)
print("Molecule:", mol)

