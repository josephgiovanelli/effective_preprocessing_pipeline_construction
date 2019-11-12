from __future__ import print_function

import argparse

import os

from results_processors.correlation_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix, chi2test, chi2tests, save_chi2tests, \
    join_result_with_simple_meta_features, get_results
from results_processors.results_mining_utils import create_possible_categories, get_filtered_datasets, load_results, \
    aggregate_results, save_simple_results, save_grouped_by_algorithm_results, compute_summary, rich_simple_results


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=True, help="step of the pipeline to execute")
    parser.add_argument("-i", "--input", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input, args.output, args.pipeline

def create_directory(result_path, directory):
    result_path = os.path.join(result_path, directory)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path

def main():
    # configure environment
    input_path, result_path, pipeline = parse_args()
    categories = create_possible_categories(pipeline)
    result_path = create_directory(result_path, 'summary')
    filtered_data_sets = get_filtered_datasets()

    # load and format the results
    simple_results = load_results(input_path, filtered_data_sets)
    simple_results = rich_simple_results(simple_results, pipeline, categories)
    save_simple_results(create_directory(result_path, 'algorithms_summary'), simple_results, filtered_data_sets)

    # summarize the results
    grouped_by_algorithm_results, grouped_by_data_set_result = aggregate_results(simple_results, categories)
    summary = compute_summary(grouped_by_algorithm_results, categories)
    save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, summary)

    # compute the chi square test
    for uniform in [True, False]:
        temp_result_path = create_directory(result_path, 'chi2tests')
        tests = chi2tests(grouped_by_algorithm_results, summary, categories, uniform)
        if uniform:
            temp_result_path = create_directory(temp_result_path, 'uniform')
        else:
            temp_result_path = create_directory(temp_result_path, 'binary')
        save_chi2tests(temp_result_path, tests)

    # compute the matrix with the number of equal result per data set
    num_equal_elements_matrix = create_num_equal_elements_matrix(grouped_by_data_set_result)
    save_num_equal_elements_matrix(create_directory(result_path, 'correlations'), num_equal_elements_matrix)


    data = get_results(grouped_by_data_set_result)
    # create the correlation matrices
    for consider_just_the_order in [True, False]:
        join = join_result_with_simple_meta_features(filtered_data_sets, data, categories, consider_just_the_order)
        correlation_matrix = create_correlation_matrix(join)
        save_correlation_matrix(create_directory(result_path, 'correlations'), correlation_matrix, consider_just_the_order)

main()