from statistics import mean

import os


def find_pipeline_iterations(history):
    for iteration in history:
        if iteration['step'] == 'pipeline':
            pass
        else:
            return iteration['iteration']

def perform_pipeline_algorithm_analysis(results):
    pipelines_iterations, algorithm_iterations = [], []

    for result in results:
        results[result]['pipeline_iterations'] = find_pipeline_iterations(results[result]['history'])
        results[result]['algorithm_iterations'] = results[result]['num_iterations'] - results[result][
            'pipeline_iterations']
        pipelines_iterations.append(results[result]['pipeline_iterations'])
        algorithm_iterations.append(results[result]['algorithm_iterations'])
        #print(result, results[result]['pipeline_iterations'], results[result]['algorithm_iterations'])
    # print(min(pipelines_iterations), min(algorithm_iterations))

    scores = {}
    for result in results:
        acronym = result.split('_')[0]
        for key in [acronym, 'all']:
            if not(key in scores):
                scores[key] = {}
            scores[key][result] = []
            scores[key][result].append((0, results[result]['baseline_score']))
            for i in range(1, 51):
                scores[key][result].append((i, results[result]['history'][i - 1]['max_history_score'] // 0.0001 / 100))
            for i in range(1, 51):
                if i <= results[result]['algorithm_iterations']:
                    scores[key][result].append((50 + i,
                                           results[result]['history'][results[result]['pipeline_iterations'] + i - 1][
                                               'max_history_score'] // 0.0001 / 100))
                else:
                    scores[key][result].append((50 + i, results[result]['best_accuracy']))

    return perform_analysis(results, scores)


def perform_algorithm_analysis(results):

    scores = {}
    for result in results:
        acronym = result.split('_')[0]
        for key in [acronym, 'all']:
            if not (key in scores):
                scores[key] = {}
            scores[key][result] = []
            scores[key][result].append((0, results[result]['baseline_score']))
            for i in range(1, 101):
                if i <= results[result]['num_iterations']:
                    scores[key][result].append((i, results[result]['history'][i - 1]['max_history_score'] // 0.0001 / 100))
                else:
                    scores[key][result].append((i, results[result]['best_accuracy']))

    return perform_analysis(results, scores)

def perform_analysis(results, scores):
    scores_to_mean = {}
    outcome = {}

    for result in results:
        for i in range(0, 101):
            acronym = result.split('_')[0]
            for key in [acronym, 'all']:
                if not (key in scores_to_mean):
                    scores_to_mean[key] = []
                    outcome[key] = []
                try:
                    scores_to_mean[key][i].append(scores[key][result][i][1])
                except:
                    scores_to_mean[key].append([])
                    scores_to_mean[key][i].append(scores[key][result][i][1])

    for i in range(0, 101):
        for key in outcome.keys():
            outcome[key].append(mean(scores_to_mean[key][i]) // 0.01 / 100)

    return outcome

def save_analysis(analysis, result_path):

    with open(os.path.join(result_path, 'result.csv'), 'w') as out:
        out.write(','.join(analysis.keys()) + '\n')

    with open(os.path.join(result_path, 'result.csv'), 'a') as out:
        for i in range(0, 101):
            row = []
            for key in analysis.keys():
                row.append(str(analysis[key][i]))
            out.write(','.join(row) + '\n')
