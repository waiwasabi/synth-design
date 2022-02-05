import numpy as np
import pandas as pd

from src.models.sac_agent import Agent
from src.models.utils import plot_learning_curve
from environment import MolSynthPool
from datetime import datetime
from os import path, makedirs

if __name__ == '__main__':
    mol_path = '../../Data/SAC/chemdiv_bb.csv'
    react_path = '../../Data/SAC/hartenfeller-smirks.csv'
    target = 'Cc1cc(C)n(-c2nc(N3CCOCC3)nc(N3CCOCC3)n2)n1'
    env = MolSynthPool(mol_path, react_path, target, max_t=30, num_reactions=58)
    agent = Agent(input_dims=[env.fp_nbits], env=env,
                  n_actions=env.num_reactants)
    n_games = 300

    reward_img_file = 'sac_reward.png'
    reward_data_file = 'sac_reward.csv'
    graph_file = 'graph.json'
    run_datetime = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    save_dir = path.join('logs/', run_datetime)
    makedirs(save_dir)

    best_score = -1000  # TODO: implement env.reward_range
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
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

        if avg_score > best_score:
            best_score = avg_score
            env.save_graph(path.join(save_dir, graph_file))
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, path.join(save_dir, reward_img_file))
        pd.DataFrame({'Reward': score_history})
