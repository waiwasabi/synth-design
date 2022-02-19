from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdChemReactions import ReactionToSmarts
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.Draw import MolToFile
from os.path import join
import re


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
        return node, 0

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

    def draw_prods(self, save_dir):
        for i, d in enumerate(self.get_graphs()):
            smiles = d['product']
            MolToFile(MolFromSmiles(smiles), join(save_dir, str(i) + '.png'))
