import pandas as pd
import chemplot as cp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('../../Data/Zinc/preprocessed/zinc_clean_ph7.csv')
    plotter = cp.Plotter.from_smiles(df["SMILES"].sample(60000))
    plotter.tsne()
    plotter.visualize_plot()
    plt.show()
