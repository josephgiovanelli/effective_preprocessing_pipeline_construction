import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = {}
    data[0] = {'title': r'$T_1$ = Features, $T_2$ = Rebalance', 'data': pd.read_csv('../results/pipeline/features_rebalance/summary/algorithms_summary/summary.csv')}
    data[1] = {'title': r'$T_1$ = Features, $T_2$ = Normalize', 'data': pd.read_csv('../results/pipeline/features_normalizer/summary/algorithms_summary/summary.csv')}
    data[2] = {'title': r'$T_1$ = Discretize, $T_2$ = Features', 'data': pd.read_csv('../results/pipeline/discretize_features/summary/algorithms_summary/summary.csv')}
    data[3] = {'title': r'$T_1$ = Discretize, $T_2$ = Rebalance', 'data': pd.read_csv('../results/pipeline/discretize_rebalance/summary/algorithms_summary/summary.csv')}
    labels = [r'$T_1$', r'$T_2$', r'$T_1$ or $T_2$', r'$T_1 \to T_2$', r'$T_2 \to T_1$', 'draw', 'baseline']
    print(data[0]['data'])


    fig, axs = plt.subplots(2, 2)
    n_groups = 3

    for i in range(0, 2):
        for j in range(0, 2):
            #fig2, ax = axs[i, j].subplots()
            index = np.arange(n_groups)
            bar_width = 0.35

            for k in range(1, 8):
                axs[i, j].bar((index * bar_width * 10) + (bar_width * (k - 1)), data[i * 2 + j]['data'].iloc[:-1, k], bar_width, label=labels[k - 1])

            axs[i, j].set(xlabel='Algorithms', ylabel='Number of wins')
            axs[i, j].set_title(data[i * 2 + j]['title'])
            plt.setp(axs, xticks=(index * bar_width * 10) + 1.05, xticklabels=['KNN', 'RF', 'NB'])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', fontsize='x-large', ncol = 8)
    fig.set_size_inches(20, 10, forward=True)
    fig.tight_layout(pad=6.0)
    fig.savefig('../results/graph.pdf')

main()