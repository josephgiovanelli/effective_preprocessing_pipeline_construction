from __future__ import print_function

from preprocessing_impact_utils import perform_algorithm_pipeline_analysis, save_analysis
from results_cooking_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix, chi2tests, save_chi2tests, join_result_with_simple_meta_features, \
    get_results, modify_class
from results_extraction_utils import create_possible_categories, get_filtered_datasets, \
    extract_results, save_results, load_results
from utils import create_directory

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-ip", "--input_pipeline", nargs="?", type=str, required=True, help="path of first input")
    parser.add_argument("-ia", "--input_algorithm", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input_pipeline, args.input_algorithm, args.output

def main():
    input_pipeline, input_algorithm, result_path = parse_args()
    result_path = create_directory(result_path, 'summary')
    filtered_data_sets = [1461]

    pipeline_algorithm_results = load_results(input_pipeline, filtered_data_sets)
    #algorithm_results = load_results(input_algorithm, filtered_data_sets)

    pipeline_algorithm_analysis = perform_algorithm_pipeline_analysis(pipeline_algorithm_results)
    #algorithm_analysis = perform_algorithm_analysis(algorithm_results)

    save_analysis(pipeline_algorithm_analysis, create_directory(result_path, 'algorithm_pipeline'))
    #save_analysis(algorithm_analysis, create_directory(result_path, 'algorithm'))
    
    # print(pipeline_algorithm_analysis)
    # print(algorithm_analysis)

    # print(','.join(pipeline_algorithm_analysis.keys()))
    # for key, value in pipeline_algorithm_analysis.items():
    #     print(key, value)








main()