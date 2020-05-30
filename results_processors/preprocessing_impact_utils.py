from statistics import mean

import os

import numpy as np
import matplotlib.pyplot as plt



def find_pipeline_iterations(history):
    for iteration in history:
        if iteration['step'] == 'algorithm':
            pass
        else:
            return iteration['iteration']

def perform_algorithm_pipeline_analysis(results):
    pipelines_iterations, algorithm_iterations = [], []

    for result in results:
        results[result]['algorithm_iterations'] = find_pipeline_iterations(results[result]['history'])
        results[result]['pipeline_iterations'] = results[result]['num_iterations'] - results[result]['algorithm_iterations']
        algorithm_iterations.append(results[result]['algorithm_iterations'])
        pipelines_iterations.append(results[result]['pipeline_iterations'])
        print(result, results[result]['pipeline_iterations'], results[result]['algorithm_iterations'])
    # print(min(pipelines_iterations), min(algorithm_iterations))

    scores = {}
    for result in results:
        acronym = result.split('_')[0]
        if not(acronym in scores):
            scores[acronym] = {}
        scores[acronym][result] = []
        scores[acronym][result].append((0, results[result]['baseline_score']))
        max_score = results[result]['baseline_score']
        for i in range(1, 51):
            if i <= results[result]['algorithm_iterations']:
                scores[acronym][result].append((i,results[result]['history'][i - 1]['max_history_score'] // 0.0001 / 100))
                max_score = results[result]['history'][i - 1]['max_history_score'] // 0.0001 / 100
            else:
                scores[acronym][result].append((i, max_score))
        for i in range(1, 51):
            scores[acronym][result].append((50 + i, results[result]['history'][results[result]['algorithm_iterations'] + i - 1]['max_history_score'] // 0.0001 / 100))

    return perform_analysis(results, scores)

def perform_analysis(results, scores):
    scores_to_kpi = {}
    outcome = {}

    for result in results:
        for i in range(0, 101):
            acronym = result.split('_')[0]
            if not (acronym in scores_to_kpi):
                scores_to_kpi[acronym] = []
                outcome[acronym] = []
            try:
                scores_to_kpi[acronym][i].append(scores[acronym][result][i][1]  / results[result]['baseline_score'])
            except:
                scores_to_kpi[acronym].append([])
                scores_to_kpi[acronym][i].append(scores[acronym][result][i][1]  / results[result]['baseline_score'])

    for i in range(1, 101):
        for key in outcome.keys():
            outcome[key].append(mean(scores_to_kpi[key][i]))
            #outcome[key].append(mean(scores_to_kpi[key][i]) // 0.01 / 100)

    return outcome

def save_analysis(analysis, result_path):

    with open(os.path.join(result_path, 'result_with_impact.csv'), 'w') as out:
        out.write(','.join(analysis.keys()) + '\n')

    with open(os.path.join(result_path, 'result_with_impact.csv'), 'a') as out:
        for i in range(0, 100):
            row = []
            for key in analysis.keys():
                row.append(str(analysis[key][i]))
            out.write(','.join(row) + '\n')

    x = np.linspace(0, 100, 100)

    plt.plot(x, analysis['nb'], label='NV', linewidth=2.5, color='lightcoral')
    plt.plot(x, analysis['knn'], label='KNN', linewidth=2.5, color='darkturquoise')
    plt.plot(x, analysis['rf'], label='RF', linewidth=2.5, color='violet')
    plt.xlabel('Configurations visited')
    plt.ylabel('Improvement score in terms of predictive accuracy')
    plt.title("Optimization on bank-marketing data-set")
    plt.legend()
    plt.xlim(0, 100)
    plt.axvline(x=50, color='#aaaaaa', linestyle='--')
    plt.grid(False)
    plt.tick_params(axis ='both', which ='both', length = 5, color='#aaaaaa')
    plt.xticks(np.linspace(0, 100, 11))
    fig = plt.gcf()
    fig.set_size_inches(10, 5, forward=True)
    fig.savefig(os.path.join(result_path, 'result_with_impact.pdf'))
