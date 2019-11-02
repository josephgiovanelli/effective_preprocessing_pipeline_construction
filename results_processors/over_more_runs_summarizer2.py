from __future__ import print_function

import argparse

import os

from results_processors.correlation_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix
from results_processors.results_mining_utils import create_possible_categories, get_filtered_datasets, load_results, \
    save_grouped_by_algorithm_results, grouped_by_dataset_to_grouped_by_algorithm, merge_dict, save_simple_results, \
    aggregate_results, compose_pipeline, check_validity, compute_result


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=True, help="step of the pipeline to execute")
    parser.add_argument("-n", "--num_runs", nargs="?", type=int, required=True, help="number of runs")
    parser.add_argument("-a", "--no_algorithms", nargs='?', type=str2bool,  required=True, default=False, help="choose if consider the algorithm or not")
    parser.add_argument("-i", "--input", nargs="?", type=str, required=True, help="path of the input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input, args.output, args.pipeline, args.num_runs, args.no_algorithms

def adapt_environment(result_path, no_algorithms):
    if no_algorithms:
        result_path = os.path.join(result_path, '4_majority_no_considering_algorithms')
    else:
        result_path = os.path.join(result_path, '4_majority_considering_algorithms')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path

def get_best_results_over_more_runs(simple_results):
    simple_results = merge_dict(simple_results)
    for key, runs in simple_results.items():
        best_accuracies = [runs[0][0]['accuracy'], runs[0][1]['accuracy']]
        best_elements = [runs[0][0], runs[0][1]]
        for i in range(1, len(runs)):
            for j in range(0, len(runs[i])):
                if runs[i][j]['accuracy'] > best_accuracies[j]:
                    best_accuracies[j] = runs[i][j]['accuracy']
                    best_elements[j] = runs[i][j]
        simple_results[key] = best_elements
    return simple_results

def get_best_results_over_more_algorithms(simple_results):
    grouped_by_algorithm_result = {}
    grouped_by_dataset_result = []

    for key, value in simple_results.items():
        acronym = key.split("_")[0]
        dataset = key.split("_")[1]

        if not(grouped_by_algorithm_result.__contains__(acronym)):
            grouped_by_algorithm_result[acronym] = {}

        grouped_by_algorithm_result[acronym][dataset] = value

    for acronym in grouped_by_algorithm_result.keys():
        grouped_by_dataset_result.append(grouped_by_algorithm_result[acronym])

    return get_best_results_over_more_runs(grouped_by_dataset_result)

def calculate_results(simple_results, pipeline, categories):
    grouped_by_dataset_result = {}
    grouped_by_algorithm_results = {}
    acronym = 'noalgorithm'

    for dataset, value in simple_results.items():

        pipelines = compose_pipeline(value[0]['pipeline'], value[1]['pipeline'], pipeline)
        result = -1
        if value[0]['accuracy'] > value[1]['accuracy']:
            result = 1
        elif value[0]['accuracy'] == value[1]['accuracy']:
            result = 0
        elif value[0]['accuracy'] < value[1]['accuracy']:
            result = 2
        if result == -1:
            raise ValueError('A very bad thing happened.')

        validity, problem = check_validity(pipelines, result)

        if not(grouped_by_dataset_result.__contains__(dataset)):
            grouped_by_dataset_result[dataset] = {}

        if not(grouped_by_algorithm_results.__contains__(acronym)):
            grouped_by_algorithm_results[acronym] = {}
            for _, category in categories.items():
                grouped_by_algorithm_results[acronym][category] = 0

        if validity:
            grouped_by_algorithm_results, grouped_by_dataset_result = compute_result(
                result, dataset, acronym, grouped_by_algorithm_results, grouped_by_dataset_result, pipelines, categories,
                [value[0]['baseline_score'], value[1]['baseline_score']], [value[0]['accuracy'], value[1]['accuracy']])
        else:
            grouped_by_dataset_result[dataset][acronym] = {'result': categories[problem], 'accuracy': value[result - 1 if result != 0 else result]['accuracy']}

    for dataset, value in grouped_by_dataset_result.items():
        for algorithm, v in value.items():
            grouped_by_dataset_result[dataset][algorithm] = v['result']
    return grouped_by_dataset_result



def main():
    input_path, result_path, pipeline, num_runs, no_algorithms = parse_args()
    categories = create_possible_categories(pipeline)

    result_path = adapt_environment(result_path, no_algorithms)

    filtered_datasets = get_filtered_datasets()

    simple_results = []
    for i in range(1, num_runs + 1):
        input = os.path.join(input_path, "run" + str(i))
        simple_results.append(load_results(input, filtered_datasets))

    simple_results = get_best_results_over_more_runs(simple_results)

    if not no_algorithms:
        save_simple_results(result_path, simple_results, filtered_datasets)

        grouped_by_algorithm_results, grouped_by_dataset_result = aggregate_results(simple_results, pipeline, categories)
        save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, categories)

        num_equal_elements_matrix = create_num_equal_elements_matrix(grouped_by_dataset_result)
        save_num_equal_elements_matrix(result_path, num_equal_elements_matrix)

        for consider_just_the_order in [True, False]:
            correlation_matrix = create_correlation_matrix(filtered_datasets, grouped_by_dataset_result, categories, consider_just_the_order)
            save_correlation_matrix(result_path, correlation_matrix, consider_just_the_order)
    else:
        grouped_by_dataset_result = get_best_results_over_more_algorithms(simple_results)
        grouped_by_dataset_result = calculate_results(grouped_by_dataset_result, pipeline, categories)

        grouped_by_algorithm_results = grouped_by_dataset_to_grouped_by_algorithm(grouped_by_dataset_result, categories, no_algorithms)
        save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, categories, no_algorithms)

main()