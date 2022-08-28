from src.chem.binary_tree import BinaryTree
from src.chem.datasets import *
import json
import pandas as pd


class SynthesisFramework:
    def __init__(self, target, objectives, weights, building_blocks: BuildingBlockDataset,
                 reactions, nbits, max_t, max_depth):
        self.target = MolWrapper(target)
        self.objectives = objectives
        self.weights = weights
        self.building_blocks = building_blocks
        self.reactions = reactions
        self.nbits = nbits
        self.max_t = max_t
        self.max_depth = max_depth
        self.props = pd.DataFrame({'SMILES': [], 'Objective': []})
        self.t = 0
        self.depth = 0
        self.state = None

    def reset(self):
        self.state = None
        self.t = 0
        self.depth = 0
        return np.zeros(self.nbits)

    def step(self, action):
        self.t += 1
        node = self.building_blocks.softmax_select(action)

        success = 0
        if self.state is None:
            self.state = BinaryTree(node, self.reactions)

        else:
            success = self.state.add(node)
            self.depth += 1

        root = self.state.root
        fp = root.morgan(self.nbits)

        props = [objective.score(root.smiles) for objective in self.objectives]
        reward = sum([weight*prop for weight, prop in zip(self.weights, props)])

        if success:
            self.props.loc[len(self.props)] = [root.smiles, *props]

        return fp, success*reward, self.terminate(), None

    def terminate(self):
        return self.t >= self.max_t or self.depth >= self.max_depth

    def get_graph(self):
        return self.state.root.as_dict()

    def save(self, path):
        with open(join(path, 'routes.json'), 'w') as f:
            json.dump(self.get_graph(), f, indent=4)
        self.state.root.draw(path)
        self.props.to_csv(join(path, 'props.csv'), index=False)


if __name__ == '__main__':
    pass
