from __future__ import print_function

import argparse

from results_processors.results_extraction_utils import create_possible_categories, get_filtered_datasets, load_results, \
    rich_simple_results, load_algorithm_results, merge_results, save_comparison, save_summary
from results_processors.utils import create_directory

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=True, help="step of the pipeline to execute")
    parser.add_argument("-ip", "--input_pipeline", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-ia", "--input_algorithm", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    return args.input_pipeline, args.input_algorithm, args.output, args.pipeline

def main():
    # configure environment
    input_path_pipeline, input_path_algorithm, result_path, pipeline = parse_args()
    categories = create_possible_categories(pipeline)
    result_path = create_directory(result_path, '_'.join(pipeline))
    filtered_data_sets = get_filtered_datasets()

    pipeline_results = load_results(input_path_pipeline, filtered_data_sets)
    pipeline_results = rich_simple_results(pipeline_results, pipeline, categories)

    algorithm_results = load_algorithm_results(input_path_algorithm, filtered_data_sets)

    comparison, summary = merge_results(pipeline_results, algorithm_results)

    save_comparison(comparison, result_path)
    save_summary(summary, result_path)

main()