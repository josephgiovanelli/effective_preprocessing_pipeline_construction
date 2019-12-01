from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from results_processors.results_cooking_utils import get_results, save_train_meta_learner, modify_class, \
    join_result_with_extracted_meta_features, encode_data, create_correlation_matrix, save_correlation_matrix
from results_processors.results_extraction_utils import create_possible_categories, get_filtered_datasets, \
    extract_results
from results_processors.utils import parse_args, create_directory


def main():
    input_path, result_path, pipeline = parse_args()
    categories = create_possible_categories(pipeline)
    result_path = create_directory(result_path, 'meta_learner')
    filtered_data_sets = get_filtered_datasets()

    _, _, grouped_by_data_set_result, _ = extract_results(input_path, filtered_data_sets, pipeline, categories)
    data = get_results(grouped_by_data_set_result)

    join = join_result_with_extracted_meta_features(data, impute=False)
    join = modify_class(join, categories, 'group_no_order')
    for algorithm in ['knn', 'rf', 'nb', 'all']:

        if algorithm != 'all':
            temp = join[join['algorithm'] == algorithm]
        else:
            temp = join


        columns = ['dataset', 'algorithm'] if algorithm != 'all' else ['dataset']
        temp = temp.drop(columns=columns)
        columns = list(temp.columns)
        rows = np.size(temp, 0)
        for column in columns:
            if temp[column].isnull().sum() == rows:
                temp = temp.drop(columns=[column])

        columns = np.size(temp, 1)
        print(algorithm + ' ' + str(rows) + ' ' + str(columns))

        name = 'ts_' + algorithm

        save_train_meta_learner(result_path, name, temp, group_no_order=False)

main()