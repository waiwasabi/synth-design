from src.chem.rdkit_wrappers import *
from collections import deque
from data.datasets import ReactionDataset

class Node:
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol_wrapper = MolWrapper(smiles)
        self.children = []


class BinaryTree:
    def __init__(self, root: Node, reaction_dataset: ReactionDataset):
        self.root = root
        self.reaction_dataset = reaction_dataset

    def add(self, mol_wrapper):
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            queue.extend(node.children)

            prod =


if __name__ == '__main__':
    pass
