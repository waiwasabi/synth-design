from collections import Counter
from rdkit import Chem
from tqdm import tqdm


def _get_node_embeddings(mol, features=('AtomicNum',)):
    output = []
    for atom in mol.GetAtoms():
        features_dict = {'AtomicNum': atom.GetAtomicNum()}
        output.append([features_dict[feat] for feat in features])
    return output


def atomic_distribution(series):
    atoms = []
    for mol in tqdm((Chem.MolFromSmiles(mol) for mol in series), total=series.size):
        atoms.extend(_get_node_embeddings(mol))
    return Counter(atoms)
