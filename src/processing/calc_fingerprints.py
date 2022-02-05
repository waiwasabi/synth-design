import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
import csv
from tqdm import tqdm


def has_any_query(mol, substructs):
    for substruct in substructs:
        if mol.HasSubstructMatch(MolFromSmiles(substruct)):
            return True
    return False


def calc_fingerprints(source, save_path, exclude=False, substructs=None):
    df = pd.read_csv(source, chunksize=20000)
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['SMILES', 'BitString'])
        for chunk in tqdm(df):
            for smiles in chunk['SMILES'].dropna():
                print(smiles)
                try:
                    mol = MolFromSmiles(smiles)
                    if exclude:
                        if not has_any_query(mol, substructs):
                            pass
                    writer.writerow([smiles, Morgan(mol, 2, nBits=1024).ToBitString()])
                except:
                    pass


if __name__ == '__main__':
    ex_path = pd.read_excel('../../Data/bin/exclude.xlsx')['Exclude']
    calc_fingerprints('../../Data/Zinc/preprocessed/zinc_clean_ph7.csv',
                      '../../Data/Zinc/processed/_zinc7_morgan.csv')
