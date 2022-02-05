import json
import pandas as pd


def smi_to_csv(data_path, write_path):
    with open(data_path, 'r') as f:
        s = f.read().split('\n')
        s = map(lambda x: x[:-13], s)

    with open(write_path, 'wb') as f:
        df = pd.DataFrame({'SMILES': s})
        df.to_csv(f, index=False)


def extract_smarts_from_json(data_path, write_path):
    with open(data_path) as f:
        d = json.load(f)
    df = pd.DataFrame(d)
    df = df[df['reaction_smarts'].notna()]
    df = df.drop(['reaction_id', 'necessary_reagent'], axis=1)
    df.to_csv(write_path, index=False)


if __name__ == '__main__':
    extract_smarts_from_json('../../Data/Uspto/raw/uspto-templates.json',
                             '../../Data/Uspto/preprocessed/uspto-clean.csv')
