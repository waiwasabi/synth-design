from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from rdkit.Chem import MolFromSmiles


class MolWrapper:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = None if smiles is None else MolFromSmiles(smiles)

    def __str__(self):
        return self.smiles


class ReactionWrapper:
    def __init__(self, smarts):
        self.smarts = smarts
        self.reaction = ReactionFromSmarts(smarts)
        self.reaction.Initialize()

    def run_reactants(self, reactants) -> MolWrapper:
        try:
            return MolWrapper(self.reaction.runReactants(reactants)[0])
        except:
            return MolWrapper(None)

    def __str__(self):
        return self.smarts


if __name__ == '__main__':
    r = ReactionWrapper("[C&$(C=O):1][O&H1].[N&$(N[#6])&!$(N=*)&!$([N&-])&!$(N#*)&!$([N&D3])&!$([N&D4])&!$(N[O,N])&!$(N[C,S]=[S,O,N]):2]>>[C:1][N&+0:2]")
    print(r)
