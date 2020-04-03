from statistics import mean

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
        print(result, results[result]['pipeline_iterations'], results[result]['algorithm_iterations'])
    # print(min(pipelines_iterations), min(algorithm_iterations))

    scores = {}
    for result in results:
        scores[result] = []
        scores[result].append((0, results[result]['baseline_score']))
        for i in range(1, 51):
            scores[result].append((i, results[result]['history'][i - 1]['max_history_score'] // 0.0001 / 100))
        for i in range(1, 51):
            if i <= results[result]['algorithm_iterations']:
                scores[result].append((50 + i,
                                       results[result]['history'][results[result]['pipeline_iterations'] + i - 1][
                                           'max_history_score'] // 0.0001 / 100))
            else:
                scores[result].append((50 + i, results[result]['best_accuracy']))

    scores_to_mean = []
    for i in range(0, 101):
        scores_to_mean.append([])
        for result in results:
            scores_to_mean[i].append(scores[result][i][1])

    outcome = []

    for i in range(0, 101):
        outcome.append(mean(scores_to_mean[i]) // 0.01 / 100)

    return outcome
