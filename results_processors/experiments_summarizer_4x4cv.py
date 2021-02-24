from __future__ import print_function

from results_cooking_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix, chi2test, chi2tests, save_chi2tests, \
    join_result_with_simple_meta_features, \
    get_results, modify_class
from results_extraction_utils import create_possible_categories, get_filtered_datasets, extract_results_4x4cv, save_results_4x4cv
from utils import parse_args, create_directory
import pandas as pd
import os

def main():
    # configure environment
    input_path, result_path, pipeline = parse_args()
    categories = create_possible_categories(pipeline)
    result_path = create_directory(result_path, 'summary')
    result_path = create_directory(result_path, '4x4cv')
    filtered_data_sets = get_filtered_datasets()

    summaries = extract_results_4x4cv(input_path, filtered_data_sets, pipeline, categories)

    save_results_4x4cv(create_directory(result_path, 'raw'), summaries)

    probs = [0.95, 0.97, 0.99]
    for prob in probs:
        results = pd.DataFrame()
        for batch in range(len(summaries)):
            train_frequencies = [summaries[batch]['train']['summary'][categories['first_second']], summaries[batch]['train']['summary'][categories['second_first']]]
            test_frequencies = [summaries[batch]['test']['summary'][categories['first_second']], summaries[batch]['test']['summary'][categories['second_first']]]
            critical, stat, statistic_test, alpha, p, p_value = chi2test(train_frequencies, test_frequencies, prob)
            results = results.append({'critical': critical, 'stat': stat, 'statistic_test': statistic_test, 'alpha': alpha, 'p': p, 'p_value': p_value}, ignore_index=True)
        results.to_csv(os.path.join(result_path, 'summary_' + str(prob) + '.csv'), index=False)


            

main()
