import numpy as np
import pandas as pd
from src.sac.sac_agent import Agent
from src.utils.plotter import plot_learning_curve
from src.sac.environment import MolSynthPool
from src.sac.guac_environment import SynthesisFramework
from src.chem.datasets import *
from src.chem.objectives import *
from datetime import datetime
from os import path, makedirs
from tqdm import tqdm
from config.settings import get_root


def run(fn, n):
    mol_path = path.join(get_root(), 'data/SAC/test_chemdiv_bb.csv')
    react_path = path.join(get_root(), 'data/SAC/hartenfeller-smirks.csv')
    """
    env = MolSynthPool(mol_path, react_path, target, max_t=30, num_reactions=58, num_reactants=5000)
    """

    building_blocks = BuildingBlockDataset(mol_path, 5000, 'SMILES')
    templates = ReactionDataset(react_path, 58, 'smirks')

    objectives = [fn]
    weights = [1]

    env = SynthesisFramework(None, objectives, weights, building_blocks, templates, nbits=1024, max_t=30, max_depth=6)

    agent = Agent(input_dims=[1024], env=env,
                  n_actions=building_blocks.n, batch_size=10)

    n_games = 300

    reward_img_file = 'sac_reward.png'
    reward_data_file = 'sac_reward.csv'
    reactant_data_file = 'reactants.csv'
    run_datetime = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    base_save_dir = path.join('../../data/logs/', run_datetime)
    makedirs(base_save_dir)

    best_score = 1
    top_score = 1
    score_history = []
    avg_score_history = []
    load_checkpoint = False
    save_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in tqdm(range(n_games)):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint and save_checkpoint:
                agent.save_models()
        if score > top_score and len(env.get_graph()) != 0:
            top_score = score
            save_dir = path.join(base_save_dir, str(i)+'.score'+str(round(score, 2)))
            makedirs(save_dir)
            env.save(save_dir)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, path.join(base_save_dir, reward_img_file))
        pd.DataFrame({'AvgReward': avg_score_history}).to_csv(path.join(base_save_dir, reward_data_file), index=False)
        # env.reactants.to_csv(path.join(base_save_dir, reactant_data_file), index=False)

    return list(env.props.nlargest(n, 'Objective')['SMILES'])
