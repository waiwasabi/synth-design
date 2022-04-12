import torch
import numpy as np

from src.utils.reservoir_sample import sample_file


class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = [] if children is None else children

    def get_children(self):
        return self.children


class BinaryTree:
    def __init__(self, root):
        self.root = root


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


