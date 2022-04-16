import torch
import numpy as np

from src.utils.reservoir_sample import sample_file
from src.chem.rdkit_wrappers import *
from collections import deque


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
