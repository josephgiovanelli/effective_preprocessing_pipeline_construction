import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = {}
    data[0] = {'title': r'$T_1$ = Feat. Eng., $T_2$ = Normalize', 'data': pd.read_csv('results/pipeline_construction/features_normalize/summary/algorithms_summary/summary.csv').reindex([1, 0, 2, 3])}
    data[1] = {'title': r'$T_1$ = Discretize, $T_2$ = Feat. Eng.', 'data': pd.read_csv('results/pipeline_construction/discretize_features/summary/algorithms_summary/summary.csv').reindex([1, 0, 2, 3])}
    data[2] = {'title': r'$T_1$ = Feat. Eng., $T_2$ = Rebalance', 'data': pd.read_csv('results/pipeline_construction/features_rebalance/summary/algorithms_summary/summary.csv').reindex([1, 0, 2, 3])}
    data[3] = {'title': r'$T_1$ = Discretize, $T_2$ = Rebalance', 'data': pd.read_csv('results/pipeline_construction/discretize_rebalance/summary/algorithms_summary/summary.csv').reindex([1, 0, 2, 3])}
    labels = [r'$T_1$', r'$T_2$', r'$T_1 \to T_2$', r'$T_2 \to T_1$', 'Baseline']
    colors = ['gold', 'mediumspringgreen', 'royalblue', 'sienna', 'mediumpurple', 'salmon']
    patterns = ["/", "\\", "o", "-", "x", ".", "O", "+", "*", "|"]

    SMALL_SIZE = 8
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    fig, axs = plt.subplots(2, 2)
    n_groups = 3

    print(data[0]['data'].drop(columns=['inconsistent', 'not_exec', 'not_exec_once']))

    for i in range(0, 2):
        for j in range(0, 2):
            #fig2, ax = axs[i, j].subplots()
            data[i * 2 + j]['data'].columns = list(range(0, len(data[i * 2 + j]['data'].columns)))
            data[i * 2 + j]['data'] = data[i * 2 + j]['data'].drop(columns=[3, 6])
            print(data[i * 2 + j]['data'])
            index = np.arange(n_groups)
            bar_width = 0.4

            for k in range(1, 6):
                axs[i, j].bar((index * bar_width * 8) + (bar_width * (k - 1)), data[i * 2 + j]['data'].iloc[:-1, k], bar_width, label=labels[k - 1], color=colors[k - 1], hatch=patterns[k - 1])

            axs[i, j].set(ylabel='Number of wins')
            axs[i, j].set(xlabel='Algorithms')
            axs[i, j].set_title(data[i * 2 + j]['title'])
            axs[i, j].set_ylim([0, 40])
            plt.setp(axs, xticks=(index * bar_width * 8) + 0.8, xticklabels=['NB', 'KNN', 'RF'])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol = 8, bbox_to_anchor=(0.5, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=axs[1,1].transAxes)
    fig.set_size_inches(20, 10, forward=True)
    fig.tight_layout(h_pad=3.0, w_pad=4.0)
    fig.savefig('results/pipeline_construction/experiments_results.pdf', bbox_extra_artists=(lgd,text), bbox_inches='tight')

    SMALL_SIZE = 8
    MEDIUM_SIZE = 17
    BIGGER_SIZE = 21

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axs = plt.subplots(1, 4)
    n_groups = 3

    for i in range(0, 4):
        # fig2, ax = axs[i, j].subplots()
        index = np.arange(n_groups)
        bar_width = 0.5

        valid = data[i]['data'].iloc[:-1, 0:8]
        valid["sum"] = valid.sum(axis=1)
        invalid = data[i]['data'].iloc[:-1, 8:11]
        invalid["sum"] = invalid.sum(axis=1)

        axs[i].bar(index * bar_width * 3, valid["sum"], label='valid', color = 'g')
        axs[i].bar(index * bar_width * 3, invalid["sum"], bottom=valid["sum"], label='invalid', color = 'r')

        #axs[i].bar((index * bar_width * 5) + (bar_width * (k - 1)), data[i]['data'].iloc[:-1, k], bar_width, label=labels[k - 1])

        axs[i].set(ylabel='Number of datasets')
        axs[i].set_title(data[i]['title'])
        axs[i].set_ylim([0, 60])
        plt.setp(axs, xticks=(index * bar_width * 3), xticklabels=['NB', 'KNN', 'RF'])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=2)
    lgd = fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.0))
    text = fig.text(-0.2, 1.05, "", transform=axs[3].transAxes)
    fig.set_size_inches(20, 5, forward=True)
    fig.tight_layout(w_pad=4.0)
    fig.savefig('results/pipeline_construction/experiments_validity.pdf', bbox_extra_artists=(lgd,text), bbox_inches='tight')

main()