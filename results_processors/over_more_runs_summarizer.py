from __future__ import print_function

import argparse

import os

from results_processors.correlation_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix
from results_processors.results_mining_utils import create_possible_categories, get_filtered_datasets, load_results, \
    aggregate_results, save_grouped_by_algorithm_results, merge_runs_by_dataset, \
    save_details_grouped_by_dataset_result, grouped_by_dataset_to_grouped_by_algorithm


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

def adapt_environment(result_path, categories, no_algorithms):
    if no_algorithms:
        result_path = os.path.join(result_path, 'majority_no_considering_algorithms')
    else:
        result_path = os.path.join(result_path, 'majority_considering_algorithms')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    categories['no_majority'] = 'no_majority'
    return result_path, categories

def main():
    input_path, result_path, pipeline, num_runs, no_algorithms = parse_args()
    categories = create_possible_categories(pipeline)

    filtered_datasets = get_filtered_datasets()

    grouped_by_algorithm_results = []
    grouped_by_dataset_result = []
    for i in range(1, num_runs + 1):
        input = os.path.join(input_path, "run" + str(i))
        simple_results = load_results(input, filtered_datasets)
        temp_grouped_by_algorithm_results, temp_grouped_by_dataset_result = aggregate_results(simple_results,
                                                                                                  pipeline,
                                                                                                  categories)
        grouped_by_algorithm_results.append(temp_grouped_by_algorithm_results)
        grouped_by_dataset_result.append(temp_grouped_by_dataset_result)

    result_path, categories = adapt_environment(result_path, categories, no_algorithms)

    details_grouped_by_dataset_result, grouped_by_dataset_result = merge_runs_by_dataset(grouped_by_dataset_result, no_algorithms)
    save_details_grouped_by_dataset_result(result_path, details_grouped_by_dataset_result, no_algorithms)

    grouped_by_algorithm_results = grouped_by_dataset_to_grouped_by_algorithm(grouped_by_dataset_result, categories, no_algorithms)
    save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, categories, no_algorithms)

    if not no_algorithms:
        num_equal_elements_matrix = create_num_equal_elements_matrix(grouped_by_dataset_result)
        save_num_equal_elements_matrix(result_path, num_equal_elements_matrix)

        for consider_just_the_order in [True, False]:
            correlation_matrix = create_correlation_matrix(filtered_datasets, grouped_by_dataset_result, categories, consider_just_the_order)
            save_correlation_matrix(result_path, correlation_matrix, consider_just_the_order)

main()