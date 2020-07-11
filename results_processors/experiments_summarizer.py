from __future__ import print_function

from results_cooking_utils import create_num_equal_elements_matrix, save_num_equal_elements_matrix, \
    create_correlation_matrix, save_correlation_matrix, chi2tests, save_chi2tests, \
    join_result_with_simple_meta_features, \
    get_results, modify_class
from results_extraction_utils import create_possible_categories, get_filtered_datasets, \
    extract_results, save_results
from utils import parse_args, create_directory


def main():
    # configure environment
    input_path, result_path, pipeline = parse_args()
    categories = create_possible_categories(pipeline)
    result_path = create_directory(result_path, 'summary')
    filtered_data_sets = get_filtered_datasets()

    simple_results, grouped_by_algorithm_results, grouped_by_data_set_result, summary = extract_results(input_path,
                                                                                                        filtered_data_sets,
                                                                                                        pipeline,
                                                                                                        categories)

    save_results(create_directory(result_path, 'algorithms_summary'),
                 filtered_data_sets,
                 simple_results,
                 grouped_by_algorithm_results,
                 summary)

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
    for group_no_order in [True, False]:
        join = join_result_with_simple_meta_features(filtered_data_sets, data)
        if group_no_order:
            join = modify_class(join, categories, 'group_no_order')
        correlation_matrix = create_correlation_matrix(join)
        save_correlation_matrix(create_directory(result_path, 'correlations'), 'correlation_matrix', correlation_matrix,
                                group_no_order)


main()
