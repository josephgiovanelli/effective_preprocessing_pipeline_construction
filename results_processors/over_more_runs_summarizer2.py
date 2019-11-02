from __future__ import print_function

import argparse

import os

from results_processors.correlation_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix
from results_processors.results_mining_utils import create_possible_categories, get_filtered_datasets, load_results, \
    save_grouped_by_algorithm_results, grouped_by_dataset_to_grouped_by_algorithm, merge_dict, save_simple_results, aggregate_results, compose_pipeline


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
        result_path = os.path.join(result_path, '5_majority_no_considering_algorithms')
    else:
        result_path = os.path.join(result_path, '5_majority_considering_algorithms')

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

def check_validity(pipelines):
    if pipelines["pipeline1"] == [] and pipelines["pipeline2"] == []:
        return False, 'not_exec'
    elif pipelines["pipeline1"] == [] or pipelines["pipeline2"] == []:
        return False, 'not_exec_once'
    return True, ''

def compute_result(result, pipeline, categories):
    if pipeline[0] == 'NoneType' and pipeline[1] == 'NoneType':
        return categories['baseline']
    elif pipeline[0] != 'NoneType' and pipeline[1] == 'NoneType':
        return categories['first']
    elif pipeline[0] == 'NoneType' and pipeline[1] != 'NoneType':
        return categories['second']
    elif pipeline[0] != 'NoneType' and pipeline[1] != 'NoneType':
        if result == 1:
            return categories['first_second']
        else:
            return categories['second_first']

def compute_result_in_case_of_draw(pipelines, categories):
    if pipelines["pipeline1"][0] != 'NoneType' and pipelines["pipeline1"][1] != 'NoneType' and pipelines["pipeline2"][0] != 'NoneType' and pipelines["pipeline2"][1] != 'NoneType':
        return categories['draw']
    elif (pipelines["pipeline1"][0] == 'NoneType' and pipelines["pipeline1"][1] == 'NoneType') or (pipelines["pipeline2"][0] == 'NoneType' and pipelines["pipeline2"][1] == 'NoneType'):
        return categories['baseline']
    elif (pipelines["pipeline1"][0] == 'NoneType' and pipelines["pipeline1"][1] != 'NoneType' and
          pipelines["pipeline2"][0] != 'NoneType' and pipelines["pipeline2"][1] == 'NoneType') \
          or \
         (pipelines["pipeline1"][0] != 'NoneType' and pipelines["pipeline1"][1] == 'NoneType' and
          pipelines["pipeline2"][0] == 'NoneType' and pipelines["pipeline2"][1] != 'NoneType'):
        return categories['first_or_second']
    elif pipelines["pipeline1"][0] != 'NoneType' and pipelines["pipeline2"][0] != 'NoneType':
        return categories['first']
    elif pipelines["pipeline1"][1] != 'NoneType' and pipelines["pipeline2"][1] != 'NoneType':
        return categories['second']
    ValueError('A very bad thing happened.')


def calculate_results(simple_results, pipeline, categories, no_algorithms):
    grouped_by_dataset_result = {}
    grouped_by_algorithm_results = {}

    for key, value in simple_results.items():
        if not no_algorithms:
            acronym = key.split("_")[0]
            dataset = key.split("_")[1]
        else:
            dataset = key
            acronym = 'noalgorithm'

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

        validity, problem = check_validity(pipelines)

        if not(grouped_by_dataset_result.__contains__(dataset)):
            grouped_by_dataset_result[dataset] = {}

        if not no_algorithms:
            if not(grouped_by_algorithm_results.__contains__(acronym)):
                grouped_by_algorithm_results[acronym] = {}
                for _, category in categories.items():
                    grouped_by_algorithm_results[acronym][category] = 0

        if validity:
            if result != 0:
                label = compute_result(result, pipelines["pipeline" + str(result)], categories)
            else:
                label = compute_result_in_case_of_draw(pipelines, categories)
        else:
            label = problem

        grouped_by_dataset_result[dataset][acronym] = label

        if not no_algorithms:
            grouped_by_algorithm_results[acronym][label] += 1

    return grouped_by_algorithm_results, grouped_by_dataset_result



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

        grouped_by_algorithm_results, grouped_by_dataset_result = calculate_results(simple_results, pipeline, categories, no_algorithms)
        save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, categories)

        num_equal_elements_matrix = create_num_equal_elements_matrix(grouped_by_dataset_result)
        save_num_equal_elements_matrix(result_path, num_equal_elements_matrix)

        for consider_just_the_order in [True, False]:
            correlation_matrix = create_correlation_matrix(filtered_datasets, grouped_by_dataset_result, categories, consider_just_the_order)
            save_correlation_matrix(result_path, correlation_matrix, consider_just_the_order)
    else:
        grouped_by_dataset_result = get_best_results_over_more_algorithms(simple_results)
        _, grouped_by_dataset_result = calculate_results(grouped_by_dataset_result, pipeline, categories, no_algorithms)

        grouped_by_algorithm_results = grouped_by_dataset_to_grouped_by_algorithm(grouped_by_dataset_result, categories, no_algorithms)
        save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, categories, no_algorithms)

main()