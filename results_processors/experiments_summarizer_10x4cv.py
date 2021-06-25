from __future__ import print_function

from results_cooking_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix, chi2test, chi2tests, save_chi2tests, \
    join_result_with_simple_meta_features, \
    get_results, modify_class
from results_extraction_utils import create_possible_categories, get_filtered_datasets, extract_results_10x4cv, save_results_10x4cv
from utils import parse_args, create_directory
import pandas as pd
import numpy as np
import os

def main():
    # configure environment
    input_path, result_path, pipeline = parse_args()
    categories = create_possible_categories(pipeline)
    result_path = create_directory(result_path, 'summary')
    result_path = create_directory(result_path, '10x4cv')
    filtered_data_sets = get_filtered_datasets()
    folds, repeat = 4, 10

    summaries = extract_results_10x4cv(input_path, filtered_data_sets, pipeline, categories, folds, repeat)

    save_results_10x4cv(create_directory(result_path, 'raw'), summaries)

    prob = 0.95
    results = pd.DataFrame()
    results_train = pd.DataFrame()
    results_test = pd.DataFrame()
    for batch in range(len(summaries)):
        total_train = summaries[batch]['train']['summary'][categories['first_second']] + summaries[batch]['train']['summary'][categories['second_first']]
        total_test = summaries[batch]['test']['summary'][categories['first_second']] + summaries[batch]['test']['summary'][categories['second_first']]
        train_frequencies = [summaries[batch]['train']['summary'][categories['first_second']], summaries[batch]['train']['summary'][categories['second_first']]]
        test_frequencies = [summaries[batch]['test']['summary'][categories['first_second']], summaries[batch]['test']['summary'][categories['second_first']]]
        critical, stat, statistic_test, alpha, p, p_value = chi2test(train_frequencies, test_frequencies, prob)
        critical_train, stat_train, statistic_test_train, alpha_train, p_train, p_value_train = chi2test(train_frequencies, [total_train * 0.1, total_train * 0.9], prob)
        critical_test, stat_test, statistic_test_test, alpha_test, p_test, p_value_test = chi2test(test_frequencies, [total_test * 0.1, total_test * 0.9], prob)
        results = results.append({'critical': critical, 'stat': stat, 'statistic_test': statistic_test, 'alpha': alpha, 'p': p, 'p_value': p_value}, ignore_index=True)
        results_train = results_train.append({'critical': critical_train, 'stat': stat_train, 'statistic_test': statistic_test_train, 'alpha': alpha_train, 'p': p_train, 'p_value': p_value_train}, ignore_index=True)
        results_test = results_test.append({'critical': critical_test, 'stat': stat_test, 'statistic_test': statistic_test_test, 'alpha': alpha_test, 'p': p_test, 'p_value': p_value_test}, ignore_index=True)
    results.to_csv(os.path.join(result_path, 'summary.csv'), index=False)
    results_train.to_csv(os.path.join(result_path, 'summary_train.csv'), index=False)
    results_test.to_csv(os.path.join(result_path, 'summary_test.csv'), index=False)
    
    results = pd.DataFrame()
    results_train = pd.DataFrame()
    results_test = pd.DataFrame()
    for repetition in range(repeat):
        batch_results = pd.DataFrame()
        batch_train_results = pd.DataFrame()
        batch_test_results = pd.DataFrame()
        for fold in range(folds):
            batch = repetition * folds + fold
            total_train = summaries[batch]['train']['summary'][categories['first_second']] + summaries[batch]['train']['summary'][categories['second_first']]
            total_test = summaries[batch]['test']['summary'][categories['first_second']] + summaries[batch]['test']['summary'][categories['second_first']]
            train_frequencies = [summaries[batch]['train']['summary'][categories['first_second']], summaries[batch]['train']['summary'][categories['second_first']]]
            test_frequencies = [summaries[batch]['test']['summary'][categories['first_second']], summaries[batch]['test']['summary'][categories['second_first']]]
            critical, stat, statistic_test, alpha, p, p_value = chi2test(train_frequencies, test_frequencies, prob)
            critical_train, stat_train, statistic_test_train, alpha_train, p_train, p_value_train = chi2test(train_frequencies, [total_train * 0.1, total_train * 0.9], prob)
            critical_test, stat_test, statistic_test_test, alpha_test, p_test, p_value_test = chi2test(test_frequencies, [total_test * 0.1, total_test * 0.9], prob)
            batch_results = batch_results.append({'critical': critical, 'stat': stat, 'statistic_test': statistic_test, 'alpha': alpha, 'p': p, 'p_value': p_value}, ignore_index=True)
            batch_train_results = batch_train_results.append({'critical': critical_train, 'stat': stat_train, 'statistic_test': statistic_test_train, 'alpha': alpha_train, 'p': p_train, 'p_value': p_value_train}, ignore_index=True)
            batch_test_results = batch_test_results.append({'critical': critical_test, 'stat': stat_test, 'statistic_test': statistic_test_test, 'alpha': alpha_test, 'p': p_test, 'p_value': p_value_test}, ignore_index=True)
        batch_results = batch_results.mean().round(3)
        batch_train_results = batch_train_results.mean().round(3)
        batch_test_results = batch_test_results.mean().round(3)
        results = results.append(batch_results, ignore_index=True)
        results_train = results_train.append(batch_train_results, ignore_index=True)
        results_test = results_test.append(batch_test_results, ignore_index=True)
    results.to_csv(os.path.join(result_path, 'summary_with_mean_.csv'), index=False)
    results_train.to_csv(os.path.join(result_path, 'summary_train_with_mean_.csv'), index=False)
    results_test.to_csv(os.path.join(result_path, 'summary_test_with_mean_.csv'), index=False)


            

main()
