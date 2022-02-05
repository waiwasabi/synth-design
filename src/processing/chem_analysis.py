import numpy as np
import pandas as pd
import rdkit.Chem.Fragments as F
import rdkit.Chem.Lipinski as L
import rdkit.Chem.rdMolDescriptors as D
from rdkit.Chem import MolFromSmiles as ToMol
from tqdm import tqdm


def count_fragments(smiles_series):
    gen = (ToMol(mol) for mol in smiles_series)
    frags_dict = {name.replace('fr_', ''): f for name, f in F.fns}
    frags = pd.DataFrame({frag: [0] for frag in frags_dict.keys()})
    for mol in gen:
        for f_name, fn in frags_dict.items():
            frags[f_name] += fn(mol)
    return frags


def lipinski_analysis(smiles_series, path):
    fns = ['exactmw', 'NumHBD', 'NumHBA', 'NumRotatableBonds', 'tpsa', 'CrippenClogP', 'FractionCSP3',
           'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings', 'NumAromaticHeterocycles',
           'NumAliphaticHeterocycles', 'NumSaturatedHeterocycles', 'NumHeteroatoms']
    extra = {'AromaticCarbocycles': D.CalcNumAromaticCarbocycles, 'AliphaticCarbocycles': D.CalcNumAliphaticCarbocycles,
             'SaturatedCarbocycles': D.CalcNumSaturatedCarbocycles, 'HeavyAtoms': L.HeavyAtomCount}
    gen = (ToMol(mol) for mol in smiles_series)
    properties = D.Properties(fns)
    data = []
    for mol in tqdm(gen):
        base = list(properties.ComputeProperties(mol))
        for f_name, fn in extra.items():
            base.append(fn(mol))
        data.append(base)
    head = fns
    head.extend(extra.keys())
    pd.DataFrame({key: value for key, value in zip(head, np.array(data).T)}).to_csv(path)
