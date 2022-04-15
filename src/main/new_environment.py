import torch
import numpy as np

from src.utils.reservoir_sample import sample_file
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from rdkit.Chem import MolFromSmiles
from collections import deque

class Node:
    def __init__(self, smiles, children=None):
        self.smiles = smiles
        self.mol_wrapper = MolWrapper(smiles)
        self.children = [] if children is None else children

    def get_children(self):
        return self.children


class ReactionWrapper:
    def __init__(self, smarts):
        self.smarts = smarts
        self.reaction = ReactionFromSmarts(smarts)
        self.reaction.initialize()

    def run_reactants(self, reactants):
        try:
            return self.reaction.runReactants(reactants)
        except:
            return None


class MolWrapper:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = None if smiles is None else MolFromSmiles(smiles)

    def __str__(self):
        return self.smiles


class BinaryTree:
    def __init__(self, root):
        self.root = root

    def add(self, mol_wrapper):
        pass


class BuildingBlockDataset:
    def __init__(self, path):
        self.path = path
        self.v = None

    def random_sample(self, n):
        self.v = sample_file(self.path, n)

    def calculate_fp(self, nbits):
        pass


class SynthesisFramework:
    def __init__(self, target, objectives):
        self.target = target
        self.objectives = objectives
        self.t = 0
        self.root = None

    def reset(self):
        self.root = None
        self.t = 0

    def step(self, action):
        self.t += 1


if __name__ == '__main__':
    print(MolWrapper(None))
