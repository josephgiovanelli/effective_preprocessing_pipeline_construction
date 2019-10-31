from __future__ import print_function

import argparse

import os

from results_processors.correlation_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix
from results_processors.results_mining_utils import create_possible_categories, get_filtered_datasets, load_results, \
    aggregate_results, save_simple_results, save_grouped_by_algorithm_results, merge_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=True, help="step of the pipeline to execute")
    parser.add_argument("-n", "--num_runs", nargs="?", type=int, required=True, help="number of runs")
    parser.add_argument("-i", "--input", nargs="?", type=str, required=True, help="path of the input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input, args.output, args.pipeline, args.num_runs

def main():
    input_path, result_path, pipeline, num_runs = parse_args()
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

    merged_dict = merge_dict(grouped_by_dataset_result)
    print(merged_dict)

main()