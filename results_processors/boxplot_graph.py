import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

    plt.axhline(y=0.05, color='#aaaaaa', linestyle='--')

    path = os.path.join("..", "results", "pipeline_construction")
    FN = pd.read_csv(os.path.join(path, "features_normalize", "summary", "10x4cv", "summary_with_mean_.csv"))
    DF = pd.read_csv(os.path.join(path, "discretize_features", "summary", "10x4cv", "summary_with_mean_.csv"))

    plt.boxplot([FN["p"], DF["p"]], widths=[0.3, 0.3])


    #plt.xlabel('Algorithms', labelpad=15.0)
    plt.xticks([1, 2], [r'$F \to N$', r'$D \to F$'])
    #plt.ylabel('Mean of the p-values among the 10\ndifferent runs of 4-folds cross validation', labelpad=15.0)
    plt.ylabel('Means of the p-values', labelpad=15.0)
    plt.yticks([0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylim(0.0, 1.0)
    #plt.title('Evaluation of the prototype building through the proposed precedence')
    #plt.tight_layout()
    plt.tight_layout(pad=0.2)
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    fig.savefig('../results/pipeline_construction/10_times_4_folds_cv.pdf')
    plt.clf()

main()