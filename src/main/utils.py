import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    plt.clf()
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.rcParams["font.family"] = "Helvetica"
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.plot(x, running_avg, color='black', rasterized=False, linewidth=1)
    plt.savefig(figure_file)
