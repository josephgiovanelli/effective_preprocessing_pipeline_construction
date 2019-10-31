from __future__ import print_function

import argparse

from results_processors.correlation_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix
from results_processors.results_miner import create_possible_categories, get_filtered_datasets, load_results, \
    aggregate_results, save_simple_results, save_grouped_by_algorithm_results


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=True, help="step of the pipeline to execute")
    parser.add_argument("-i", "--first_input", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-ii", "--second_input", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    input_paths = [args.first_input, args.second_input]
    result_path = args.output
    pipeline = args.pipeline
    return input_paths, result_path, pipeline

def main():
    input_paths, result_path, pipeline = parse_args()
    categories = create_possible_categories(pipeline)

    filtered_datasets = get_filtered_datasets()

    simple_results = load_results(input_paths, filtered_datasets)
    save_simple_results(result_path, simple_results, filtered_datasets)

    grouped_by_algorithm_results, grouped_by_dataset_result = aggregate_results(simple_results, pipeline, categories)
    save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, categories)

    num_equal_elements_matrix = create_num_equal_elements_matrix(grouped_by_dataset_result)
    save_num_equal_elements_matrix(result_path, num_equal_elements_matrix)

    correlation_matrix = create_correlation_matrix(filtered_datasets, grouped_by_dataset_result, categories,
                                                   considerate_just_order = False)
    save_correlation_matrix(result_path, correlation_matrix)

main()