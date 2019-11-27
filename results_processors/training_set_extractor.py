from __future__ import print_function

from results_processors.results_cooking_utils import get_results, save_train_meta_learner, modify_class, \
    join_result_with_extracted_meta_features
from results_processors.results_extraction_utils import create_possible_categories, get_filtered_datasets, \
    extract_results
from results_processors.utils import parse_args, create_directory


def main():
    # configure environment
    input_path, result_path, pipeline = parse_args()
    categories = create_possible_categories(pipeline)
    result_path = create_directory(result_path, 'meta_learner')
    filtered_data_sets = get_filtered_datasets()

    _, _, grouped_by_data_set_result, _ = extract_results(input_path, filtered_data_sets, pipeline, categories)
    data = get_results(grouped_by_data_set_result)

    join = join_result_with_extracted_meta_features(data)
    join = modify_class(join, categories, 'group_no_order')
    save_train_meta_learner(result_path, join, group_no_order=True)


main()