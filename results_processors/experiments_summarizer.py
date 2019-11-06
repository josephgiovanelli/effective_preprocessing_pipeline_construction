from __future__ import print_function

import argparse

from results_processors.correlation_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix, chi2test, chi2tests, save_chi2tests
from results_processors.results_mining_utils import create_possible_categories, get_filtered_datasets, load_results, \
    aggregate_results, save_simple_results, save_grouped_by_algorithm_results, compute_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=True, help="step of the pipeline to execute")
    parser.add_argument("-i", "--input", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input, args.output, args.pipeline

def main():
    input_path, result_path, pipeline = parse_args()
    categories = create_possible_categories(pipeline)

    filtered_datasets = get_filtered_datasets()

    simple_results = load_results(input_path, filtered_datasets)
    save_simple_results(result_path, simple_results, filtered_datasets)

    grouped_by_algorithm_results, grouped_by_dataset_result = aggregate_results(simple_results, pipeline, categories)
    summary = compute_summary(grouped_by_algorithm_results, categories)
    save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, summary)

    test, order_test, not_order_test = chi2tests(grouped_by_algorithm_results, summary, categories)
    save_chi2tests(result_path, test, order_test, not_order_test)

    num_equal_elements_matrix = create_num_equal_elements_matrix(grouped_by_dataset_result)
    save_num_equal_elements_matrix(result_path, num_equal_elements_matrix)

    for consider_just_the_order in [True, False]:
        correlation_matrix = create_correlation_matrix(filtered_datasets, grouped_by_dataset_result, categories, consider_just_the_order)
        save_correlation_matrix(result_path, correlation_matrix, consider_just_the_order)

main()