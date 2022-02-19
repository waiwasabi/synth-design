from src.main.sac.hyperparameters import *
from rdkit.DataStructs.cDataStructs import CreateFromBitString, TanimotoSimilarity
from src.utils.reservoir_sample import sample_file
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from rdkit.Chem import MolFromSmiles, QED, SmilesMolSupplier
from rdkit.Chem.Fingerprints.SimilarityScreener import TopNScreener
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
import numpy as np
import pandas as pd
from src.utils.chem import SyntheticTree, Node
from os.path import join
from os import makedirs
import json
from random import shuffle


def arr_to_str(arr):
    arr = arr if type(arr) == list else arr.tolist()
    return ''.join(map(lambda x: str(int(x)), arr))


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
        self.qed_cutoff = 0.7

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

        shuffle(self.reactions)

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
        reward = RXN_FACTOR*rxn_reward * (SIM_FACTOR * sim_reward + QED_FACTOR * qed_reward)  # TODO: add more to reward
        return root_fp, reward, self.is_terminal(qed_reward,
                                                 rxn_reward), None

    def is_terminal(self, qed, rxn_reward):
        return self.t >= self.max_t or (qed > self.qed_cutoff and rxn_reward != -1)

    def top_match_from_softmax(self, arr):
        bit_vect = CreateFromBitString(arr_to_str(arr))
        supply = SmilesMolSupplier(self.mol_path)
        f_fp = lambda x: Morgan(x, 2, nBits=self.fp_nbits)
        return TopNScreener(1, probe=bit_vect, metric=TanimotoSimilarity,
                            dataSource=supply, fingerprinter=f_fp)[0][1]

    def save_graph(self, save_dir):
        graphs = self.state.get_graphs()
        jason = json.dumps(graphs)
        products = [d['product'] for d in graphs]
        qed = [QED.default(MolFromSmiles(smiles)) for smiles in products]
        with open(join(save_dir, 'routes.json'), 'w') as f:
            json.dump(jason, f)
        pd.DataFrame({'SMILES': products, 'QED': qed}).to_csv(join(save_dir, 'props.csv'))
        image_dir = join(save_dir, 'Images')
        makedirs(image_dir, exist_ok=True)
        self.state.draw_prods(image_dir)


if __name__ == '__main__':
    mol_path = '../../Data/SAC/chemdiv_bb.csv'
    react_path = '../../Data/SAC/hartenfeller-smirks.csv'
    target = 'Cc1cc(C)n(-c2nc(N3CCOCC3)nc(N3CCOCC3)n2)n1'
    env = MolSynthPool(mol_path, react_path, target, max_t=30, num_reactions=58, num_reactants=10000)
    env.reset()
