from src.utils.reservoir_sample import sample_file
from src.chem.rdkit_wrappers import *


class Dataset:
    def __init__(self, path, n, data_head):
        self.path = path
        self.data_head = data_head
        self.v = sample_file(self.path, n)[data_head]


class BuildingBlockDataset:
    def __init__(self, path):
        self.path = path
        self.v = None

    def random_sample(self, n):
        self.v = sample_file(self.path, n)


class ReactionDataset(Dataset):
    def __init__(self, path, n, data_head):
        super().__init__(path, n, data_head)
        self.v = [ReactionWrapper(smirks) for smirks in self.v]

    def find_reaction(self, mol1, mol2):
        for reaction in self.v:
            product = reaction.run_reactants([mol1, mol2])
            if product is not None:
                return product

        return MolWrapper(None)


if __name__ == '__main__':
    data = ReactionDataset()