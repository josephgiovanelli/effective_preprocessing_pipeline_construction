from __future__ import print_function

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

    for impute in [True, False]:
        join = join_result_with_extracted_meta_features(data, impute)
        join = modify_class(join, categories, 'group_no_order')
        for algorithm in ['knn', 'rf', 'nb', 'all']:

            if algorithm != 'all':
                temp = join[join['algorithm'] == algorithm]
            else:
                temp = join

            if impute:
                temp = temp.drop(columns=['class', 'algorithm'])
                columns = list(temp.columns)
                temp = pd.DataFrame(SimpleImputer(strategy="constant").fit_transform(temp), columns=columns)
                columns = ['dataset', 'algorithm', 'class'] if algorithm == 'all' else ['dataset', 'class']
                temp = pd.merge(temp, join[columns], left_on='dataset', right_on='dataset')
                temp = temp.drop(columns=['dataset'])
            else:
                temp = temp.drop(columns=['dataset', 'algorithm'])
                columns = list(temp.columns)
                for column in columns:
                    if temp[column].isnull().sum() != 0:
                        temp = temp.drop(columns=[column])

            name = 'ts_' + algorithm + ('_imputed' if impute else '')

            correlation_matrix = create_correlation_matrix(temp)
            save_correlation_matrix(create_directory(result_path, 'correlations'), name, correlation_matrix, group_no_order=False)

            save_train_meta_learner(result_path, name, temp, group_no_order=False)

main()