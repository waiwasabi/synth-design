from src.models.hyperparameters import *
from src.utils.similarity import top_n_chunked_tanimoto
from rdkit.DataStructs.cDataStructs import CreateFromBitString
from src.utils.reservoir_sample import sample_file
from rdkit.Chem.rdChemReactions import ReactionFromSmarts, ReactionToSmarts
from rdkit.Chem import MolFromSmiles, MolToSmiles, QED
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import time
import sys
from os.path import join
import json


def arr_to_str(arr):
    arr = arr if type(arr) == list else arr.tolist()
    return ''.join(map(lambda x: str(int(x)), arr))


class SyntheticRoute:
    def __init__(self, data_path):
        self.reactants_path = data_path

    def top_n_from_embed(self, raw_embed: np.array, n: int, chunksize: int):
        """
        :param raw_embed: array-like. A vector representing the fingerprint of a query molecule
        :param n: number of matches to extract
        :param chunksize: number of samples to process in memory per iteration
        :return: a list of n fingerprints with the top n similarity scores compared to the probe.
        """
        embed = arr_to_str(np.rint(raw_embed))
        out = top_n_chunked_tanimoto(n, embed, self.reactants_path, chunksize)
        return out


class Node:
    def __init__(self, smiles: str):
        assert (type(smiles) == str)
        self.v = smiles
        self.prod = None
        self.reactants = None
        self.reaction = None

    def set_prod(self, prod):
        self.prod = prod

    def set_reactants(self, reactants):
        self.reactants = reactants

    def is_root(self):
        return self.prod is None and self.reactants is not None

    def is_top(self):
        return self.prod is None

    def eval_depth(self):
        if self.reactants is None:
            return 1
        l = [node.eval_depth() for node in self.reactants]
        return 1 + max(l)

    def get_node_dict(self):
        if self.reaction is None:
            return self.v
        else:
            return {'product': self.v, 'reaction': self.reaction,
                    'reactants': [node.get_node_dict() for node in self.reactants]}


class SyntheticTree:
    def __init__(self, target: str = None, nbits=1024):
        self.root = None
        self.nodes = []
        self.target = MolFromSmiles(target)
        self.nbits = nbits
        self.target_fp = Morgan(self.target, 2, self.nbits)
        self.reaction = None
        self.no_match = []

    def match_reaction(self, rxn):
        return [n for n in self.nodes if n.is_top() and rxn.IsMoleculeReactant(MolFromSmiles(n.v))]

    def get_top_nodes(self):
        return [n for n in self.nodes if n.is_top()]

    def get_matching_reactions(self, mol, rxn_list) -> list:
        return [rxn for rxn in rxn_list if rxn.IsMoleculeReactant(mol)]

    def add_reactant(self, node, rxn_list) -> (Node, int):
        mol = MolFromSmiles(node.v)
        rxns = self.get_matching_reactions(mol, rxn_list)
        for n in reversed(self.get_top_nodes()):
            for rxn in rxns:
                try:
                    prod = rxn.RunReactants([mol, MolFromSmiles(n.v)])[0]
                    self.root = Node(MolToSmiles(prod[0]))
                    self.react([node, n], self.root, rxn)
                    self.nodes.extend([node, self.root])
                    return self.root, 1
                except (ValueError, IndexError) as e:
                    continue
            break
        self.nodes.append(node)
        self.root = node
        if len(rxns) == 0:
            self.no_match.append(node.v)
        return node, -1

    def react(self, reactants, product, reaction):
        product.set_reactants(reactants)
        product.reaction = ReactionToSmarts(reaction)
        for reactant in reactants:
            reactant.set_prod(product)

    def observe(self):
        return self.root

    def eval_depth(self):
        return max([node.eval_depth() for node in self.get_top_nodes()])

    def tanimoto_sim(self, fp):
        return TanimotoSimilarity(self.target_fp, fp)

    def get_graphs(self):
        return [node.get_node_dict() for node in self.get_top_nodes() if type(node.get_node_dict()) != str]


class MolSynthPool:
    def __init__(self, mol_path: str, react_path: str, target: str, max_t: int, num_reactants: int = 1000,
                 num_reactions: int = 150, mol_key: str = 'SMILES', rxn_key: str = 'smirks',
                 fp_nbits: int = 1024, has_fingerprints: int = True):
        self.state = None
        self.mol_path = mol_path
        self.react_path = react_path
        self.target = target
        self.max_t = max_t
        self.num_reactions = num_reactions
        self.num_reactants = num_reactants
        self.fp_nbits = fp_nbits
        self.has_fingerprints = has_fingerprints
        self.t = 0

        #  TODO: review the below
        #  TODO: implement fp_calculation
        self.reactants = sample_file(mol_path, num_reactants)
        self.react_smarts = pd.read_csv(react_path).sample(num_reactions)[rxn_key]
        self.rxn_valid_coeff = 0.1

        self.reactions = []  # reactions are kept as objects in memory. Not scalable.
        for smarts in self.react_smarts:
            rxn = ReactionFromSmarts(smarts)
            rxn.Initialize()
            self.reactions.append(rxn)

    def reset(self) -> np.array:
        """
        :param seed: random seed
        :param train: (bool) specifies whether to reset to a training state or test state.
        :return: initial observation
        """
        self.state = SyntheticTree(self.target)
        self.t = 0
        return np.zeros(self.fp_nbits)

    def step(self, action: np.array) -> (np.array, int, bool, dict):
        """
        :param action: (softmax activated array, sigmoid/tanh activated probabilities of reaction)
        :return: observation_,
        """
        self.t += 1
        mol = self.reactants.iloc[[np.argmax(action)]]
        node = Node(mol.values[0][0])
        root_node, rxn_reward = self.state.add_reactant(node, self.reactions)
        root_mol = MolFromSmiles(root_node.v)
        if root_mol is None:
            return np.zeros(self.fp_nbits), -10, True, None
        root_fp = Morgan(root_mol, 2, nBits=self.fp_nbits)
        sim_reward = self.state.tanimoto_sim(root_fp)
        qed_reward = QED.default(root_mol)
        reward = RXN_FACTOR*rxn_reward + SIM_FACTOR*sim_reward + QED_FACTOR*qed_reward # TODO: add more to reward
        return root_fp, reward, self.is_terminal(), None

    def is_terminal(self):
        return self.t >= self.max_t

    def save_graph(self, save_dir):
        temp = self.state.get_graphs()  # TODO: delete
        graphs = json.dumps(self.state.get_graphs())
        with open(save_dir, 'w') as f:
            json.dump(graphs, f)


if __name__ == '__main__':
    x = MolSynthPool('../../Data/Zinc/processed/test_zinc7_morgan.csv',
                     '../../Data/Uspto/preprocessed/uspto-clean.csv', 'CCN(CC)C(=O)COc1ccc(cc1OC)CC=C',
                     15, num_reactions=500, num_reactants=1000)
    x.reset()
