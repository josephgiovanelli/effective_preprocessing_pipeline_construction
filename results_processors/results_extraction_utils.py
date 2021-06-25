import collections
import os
import json

import pandas as pd

from os import listdir
from os.path import isfile, join

algorithms = ['RandomForest', 'NaiveBayes', 'KNearestNeighbors']
benchmark_suite = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307,
                       300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1510, 1489, 1494, 1497, 1501,
                       1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381, 40499,
                       40668, 40966, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978,
                       40670, 40701]
extended_benchmark_suite = [41145, 41156, 41157, 4541, 41158, 42742, 40498, 42734, 41162, 42733, 42732, 1596, 40981, 40685, 
                        4135, 41142, 41161, 41159, 41163, 41164, 41138, 41143, 41146, 41150, 40900, 41165, 41166, 41168, 41169, 
                        41147, 1111, 1169, 41167, 41144, 1515, 1457, 181]


def create_possible_categories(pipeline):
    first = pipeline[0][0].upper()
    second = pipeline[1][0].upper()
    first_or_second = first + 'o' + second
    first_second = first + second
    second_first = second + first
    draw = first_second + 'o' + second_first
    baseline = 'baseline'
    inconsistent = 'inconsistent'
    not_exec = 'not_exec'
    not_exec_once = 'not_exec_once'

    return {'first': first,
            'second': second,
            'first_or_second': first_or_second,
            'first_second': first_second,
            'second_first': second_first,
            'draw': draw,
            'baseline': baseline,
            'inconsistent': inconsistent,
            'not_exec': not_exec,
            'not_exec_once': not_exec_once}

def get_filtered_datasets():
    df = pd.read_csv('results_processors/meta_features/simple-meta-features.csv')
    df = df.loc[df['did'].isin(list(dict.fromkeys(benchmark_suite + extended_benchmark_suite + [10, 20, 26])))]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()

def merge_dict(list):
    ''' Merge dictionaries and keep values of common keys in list'''
    new_dict = {}
    for key, value in list[0].items():
        new_value = []
        for dict in list:
            new_value.append(dict[key])
        new_dict[key] = new_value
    return new_dict

def load_results(input_path, filtered_datasets):
    comparison = {}
    confs = [os.path.join(input_path, 'conf1'), os.path.join(input_path, 'conf2')]
    for path in confs:
        files = [f for f in listdir(path) if isfile(join(path, f))]
        results = [f[:-5] for f in files if f[-4:] == 'json']
        comparison[path] = {}
        for algorithm in algorithms:
            for dataset in filtered_datasets:
                acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
                acronym += '_' + str(dataset)
                if acronym in results:
                    with open(os.path.join(path, acronym + '.json')) as json_file:
                        data = json.load(json_file)
                        accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                        pipeline = str(data['context']['best_config']['pipeline']).replace(' ', '').replace(',', ' ')
                        num_iterations = data['context']['iteration'] + 1
                        best_iteration = data['context']['best_config']['iteration'] + 1
                        baseline_score = data['context']['baseline_score'] // 0.0001 / 100
                else:
                    accuracy = 0
                    pipeline = ''
                    num_iterations = 0
                    best_iteration = 0
                    baseline_score = 0

                comparison[path][acronym] = {}
                comparison[path][acronym]['accuracy'] = accuracy
                comparison[path][acronym]['baseline_score'] = baseline_score
                comparison[path][acronym]['num_iterations'] = num_iterations
                comparison[path][acronym]['best_iteration'] = best_iteration
                comparison[path][acronym]['pipeline'] = pipeline

    return dict(collections.OrderedDict(sorted(merge_dict([comparison[confs[0]], comparison[confs[1]]]).items())))

def load_algorithm_results(input_path, filtered_datasets):
    results_map = {}
    files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    results = [f[:-5] for f in files if f[-4:] == 'json']
    for algorithm in algorithms:
        for dataset in filtered_datasets:
            acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
            acronym += '_' + str(dataset)
            if acronym in results:
                with open(os.path.join(input_path, acronym + '.json')) as json_file:
                    data = json.load(json_file)
                    accuracy = data['context']['best_config']['score'] // 0.0001 / 100
                    pipeline = str(data['context']['best_config']['pipeline']).replace(' ', '').replace(',', ' ')
                    num_iterations = data['context']['iteration'] + 1
                    best_iteration = data['context']['best_config']['iteration'] + 1
                    baseline_score = data['context']['baseline_score'] // 0.0001 / 100
            else:
                accuracy = 0
                pipeline = ''
                num_iterations = 0
                best_iteration = 0
                baseline_score = 0

            results_map[acronym] = {}
            results_map[acronym]['accuracy'] = accuracy
            results_map[acronym]['baseline_score'] = baseline_score
            results_map[acronym]['num_iterations'] = num_iterations
            results_map[acronym]['best_iteration'] = best_iteration
            results_map[acronym]['pipeline'] = pipeline

    return results_map


def save_simple_results(result_path, simple_results, filtered_datasets):
    def values_to_string(values):
        return [str(value).replace(',', '') for value in values]

    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        if os.path.exists('{}.csv'.format(acronym)):
            os.remove('{}.csv'.format(acronym))
        with open(os.path.join(result_path, '{}.csv'.format(acronym)), 'w') as out:
            first_element = simple_results[list(simple_results.keys())[0]]
            conf_keys = first_element['conf1'].keys()
            conf1_header = ','.join([a + '1' for a in conf_keys])
            conf2_header = ','.join([a + '2' for a in conf_keys])
            result_header = ','.join(first_element['result'].keys())
            header = ','.join([result_header, conf1_header, conf2_header])
            out.write('dataset,name,dimensions,' + header + '\n')


    df = pd.read_csv('results_processors/meta_features/simple-meta-features.csv')
    df = df.loc[df['did'].isin(filtered_datasets)]

    for key, value in simple_results.items():
        acronym = key.split('_')[0]
        data_set = key.split('_')[1]
        name = df.loc[df['did'] == int(data_set)]['name'].values.tolist()[0]
        dimensions = ' x '.join([str(int(a)) for a in df.loc[df['did'] == int(data_set)][
            ['NumberOfInstances', 'NumberOfFeatures']].values.flatten().tolist()])


        with open(os.path.join(result_path, '{}.csv'.format(acronym)), 'a') as out:
            results = ','.join(values_to_string(value['result'].values()))
            conf1 = ','.join(values_to_string(value['conf1'].values()))
            conf2 = ','.join(values_to_string(value['conf2'].values()))
            row = ','.join([data_set, name, dimensions, results, conf1, conf2])
            out.write(row + '\n')


def compose_pipeline(pipeline1, pipeline2, scheme):
    pipelines = {'pipeline1': [], 'pipeline2': []}
    parameters = {'pipeline1': [], 'pipeline2': []}
    for step in scheme:
        if pipeline1 != '':
            raw_pipeline1 = json.loads(pipeline1.replace('\'', '\"').replace(' ', ',').replace('True', '1').replace('False', '0'))
            pipelines['pipeline1'].append(raw_pipeline1[step][0].split('_')[1])
            for param in raw_pipeline1[step][1]:
                parameters['pipeline1'].append(raw_pipeline1[step][1][param])
        if pipeline2 != '':
            raw_pipeline2 = json.loads(pipeline2.replace('\'', '\"').replace(' ', ',').replace('True', '1').replace('False', '0'))
            pipelines['pipeline2'].append(raw_pipeline2[step][0].split('_')[1])
            for param in raw_pipeline2[step][1]:
                parameters['pipeline2'].append(raw_pipeline2[step][1][param])
    return pipelines, parameters

def have_same_steps(pipelines):
    pipeline1_has_first = not pipelines['pipeline1'][0].__contains__('NoneType')
    pipeline1_has_second = not pipelines['pipeline1'][1].__contains__('NoneType')
    pipeline2_has_first = not pipelines['pipeline2'][0].__contains__('NoneType')
    pipeline2_has_second = not pipelines['pipeline2'][1].__contains__('NoneType')
    both_just_first =  pipeline1_has_first and not pipeline1_has_second and pipeline2_has_first and not pipeline2_has_second
    both_just_second =  not pipeline1_has_first and pipeline1_has_second and not pipeline2_has_first and pipeline2_has_second
    both_baseline =  not pipeline1_has_first and not pipeline1_has_second and not pipeline2_has_first and not pipeline2_has_second
    return both_just_first or both_just_second or both_baseline

def check_validity(pipelines, result, acc1, acc2):
    if pipelines['pipeline1'] == [] and pipelines['pipeline2'] == []:
        validity, problem = False, 'not_exec'
    elif pipelines['pipeline1'] == [] or pipelines['pipeline2'] == []:
        validity, problem = False, 'not_exec_once'
    else:
        if pipelines['pipeline1'].__contains__('NoneType') and pipelines['pipeline2'].__contains__('NoneType'):
            validity = result == 0
        elif pipelines['pipeline1'].__contains__('NoneType') and not(pipelines['pipeline2'].__contains__('NoneType')):
            validity = result == 0 or result == 2
        elif not(pipelines['pipeline1'].__contains__('NoneType')) and pipelines['pipeline2'].__contains__('NoneType'):
            validity = result == 0 or result == 1
        else:
            validity = True
        problem = '' if validity else 'inconsistent'

    if not(validity) and pipelines['pipeline1'] != [] and pipelines['pipeline2'] != []:
        if have_same_steps(pipelines):
            validity, problem, result = True, '', 0

    return validity, problem, result


def compute_result(result, pipelines, categories, baseline_scores, scores):
    if baseline_scores[0] != baseline_scores[1]:
        raise Exception('Baselines with different scores')

    #case a, b, c, e, i
    if result == 0 and (baseline_scores[0] == scores[0] or baseline_scores[1] == scores[1]):
        return 'baseline'
    #case d, o
    elif pipelines['pipeline1'].count('NoneType') == 2 or pipelines['pipeline2'].count('NoneType') == 2:
        if pipelines['pipeline1'].count('NoneType') == 2 and pipelines['pipeline2'].count('NoneType') == 0:
            if result == 2:
                return categories['second_first']
            else:
                raise Exception('pipeline2 is not winning. ' + str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores))
        elif pipelines['pipeline1'].count('NoneType') == 0 and pipelines['pipeline2'].count('NoneType') == 2:
            if result == 1:
                return categories['first_second']
            else:
                raise Exception('pipeline1 is not winning. ' + str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores))
        else:
            raise Exception('Baseline doesn\'t draw with a pipeline with just one operation. pipelines:' + str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores))
    #case f, m, l, g
    elif pipelines['pipeline1'].count('NoneType') == 1 and pipelines['pipeline2'].count('NoneType') == 1:
        #case f
        if pipelines['pipeline1'][0] == 'NoneType' and pipelines['pipeline2'][0] == 'NoneType':
            if result == 0:
                return categories['second']
            else:
                raise Exception('pipelines is not drawing. ' + str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores))
        #case m
        elif pipelines['pipeline1'][1] == 'NoneType' and pipelines['pipeline2'][1] == 'NoneType':
            if result == 0:
                return categories['first']
            else:
                raise Exception('pipelines is not drawing. ' + str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
        #case g, l
        elif (pipelines['pipeline1'][0] == 'NoneType' and pipelines['pipeline2'][1] == 'NoneType') or (pipelines['pipeline1'][1] == 'NoneType' and pipelines['pipeline2'][0] == 'NoneType'):
            if result == 0:
                return categories['first_or_second']
            else:
                raise Exception('pipelines is not drawing. ' + str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
    #case h, n
    elif pipelines['pipeline1'].count('NoneType') == 1:
        #case h
        if pipelines['pipeline1'][0] == 'NoneType':
            if result == 0:
                return categories['second']
            elif result == 2:
                return categories['second_first']
            else:
                raise Exception('pipeline2 is not winning. ' + str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
        #case n
        elif pipelines['pipeline1'][1] == 'NoneType':
            if result == 0:
                return categories['first']
            elif result == 2:
                return categories['second_first']
            else:
                raise Exception('pipeline2 is not winning. ' + str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
    # case p, q
    elif pipelines['pipeline2'].count('NoneType') == 1:
        # case p
        if pipelines['pipeline2'][0] == 'NoneType':
            if result == 0:
                return categories['second']
            elif result == 1:
                return categories['first_second']
            else:
                raise Exception('pipeline1 is not winning. ' + str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
        # case q
        elif pipelines['pipeline2'][1] == 'NoneType':
            if result == 0:
                return categories['first']
            elif result == 1:
                return categories['first_second']
            else:
                raise Exception('pipeline1 is not winning. ' + str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')
    #case r
    elif pipelines['pipeline1'].count('NoneType') == 0 and pipelines['pipeline2'].count('NoneType') == 0:
        if result == 0:
            return categories['draw']
        elif result == 1:
            return categories['first_second']
        elif result == 2:
            return categories['second_first']
    else:
        raise Exception('This configuration matches nothing. ' + str(pipelines) + ' baseline_score ' + str(baseline_scores[0]) + ' scores ' + str(scores) + ' algorithm ')


def instantiate_results(grouped_by_dataset_result, grouped_by_algorithm_results, dataset, acronym, categories):
    if not (grouped_by_dataset_result.__contains__(dataset)):
        grouped_by_dataset_result[dataset] = {}

    if not (grouped_by_algorithm_results.__contains__(acronym)):
        grouped_by_algorithm_results[acronym] = {}
        for _, category in categories.items():
            grouped_by_algorithm_results[acronym][category] = 0

def get_winner(accuracy1, accuracy2):
    if accuracy1 > accuracy2:
        return 1
    elif accuracy1 == accuracy2:
        return 0
    elif accuracy1 < accuracy2:
        return 2
    else:
        raise ValueError('A very bad thing happened.')

def rich_simple_results(simple_results, pipeline_scheme, categories):
    for key, value in simple_results.items():
        first_configuration = value[0]
        second_configuration = value[1]
        pipelines, parameters = compose_pipeline(first_configuration['pipeline'], second_configuration['pipeline'], pipeline_scheme)

        try:
            winner = get_winner(first_configuration['accuracy'], second_configuration['accuracy'])
        except Exception as e:
            print(str(e))

        validity, label, winner = check_validity(pipelines, winner, first_configuration['accuracy'], second_configuration['accuracy'])

        if validity:
            try:
                baseline_scores = [first_configuration['baseline_score'], second_configuration['baseline_score']]
                accuracies = [first_configuration['accuracy'], second_configuration['accuracy']]
                label = compute_result(winner, pipelines, categories, baseline_scores, accuracies)
            except Exception as e:
                print(str(e))
                label = categories['inconsistent']

        first_configuration['pipeline'] = pipelines['pipeline1']
        second_configuration['pipeline'] = pipelines['pipeline2']

        first_configuration['parameters'] = parameters['pipeline1']
        second_configuration['parameters'] = parameters['pipeline2']

        value.append({'winner': winner, 'validity': validity, 'label': label})
        simple_results[key] = {'conf1': value[0], 'conf2': value[1], 'result': value[2]}
    return simple_results

def aggregate_results(simple_results, categories):
    grouped_by_dataset_result = {}
    grouped_by_algorithm_results = {}

    for key, value in simple_results.items():
        acronym = key.split('_')[0]
        data_set = key.split('_')[1]
        instantiate_results(grouped_by_dataset_result, grouped_by_algorithm_results, data_set, acronym, categories)

        grouped_by_dataset_result[data_set][acronym] = value['result']['label']
        grouped_by_algorithm_results[acronym][value['result']['label']] += 1

    return grouped_by_algorithm_results, grouped_by_dataset_result


def compute_summary(grouped_by_algorithm_results, categories):
    summary = {}
    for _, category in categories.items():
        summary[category] = sum(x[category] for x in grouped_by_algorithm_results.values())
    return summary

def save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, summary, no_algorithms = False):
    with open(os.path.join(result_path, 'summary.csv'), 'w') as out:
        out.write(',' + ','.join(summary.keys()) + '\n')
        if not(no_algorithms):
            for key, value in grouped_by_algorithm_results.items():
                row = key
                for k, v in value.items():
                    row += ',' + str(v)
                row += '\n'
                out.write(row)
        row = 'summary'
        for key, value in summary.items():
            row += ',' + str(value)
        out.write(row)

def save_details_grouped_by_dataset_result(result_path, details_grouped_by_dataset_result):
    for element in algorithms:
        header = False
        acronym = ''.join([a for a in element if a.isupper()]).lower()

        with open(os.path.join(result_path, '{}.csv'.format(acronym)), 'w') as out:

            for dataset, detail_results in details_grouped_by_dataset_result.items():
                if not(header):
                    out.write(',' + ','.join(detail_results[acronym].keys()) + '\n')
                    header = True
                results = ','.join(list(str(elem).replace(',', '')
                                                 .replace('[', '')
                                                 .replace(']', '')
                                                 .replace('\'', '') for elem in detail_results[acronym].values()))
                out.write(dataset + ',' + results + '\n')

def extract_results(input_path, filtered_data_sets, pipeline, categories):
    # load and format the results
    simple_results = load_results(input_path, filtered_data_sets)
    simple_results = rich_simple_results(simple_results, pipeline, categories)

    # summarize the results
    grouped_by_algorithm_results, grouped_by_data_set_result = aggregate_results(simple_results, categories)
    summary = compute_summary(grouped_by_algorithm_results, categories)

    return simple_results, grouped_by_algorithm_results, grouped_by_data_set_result, summary

def compute_summary_from_data_set_results(dataset_results, categories):
    summary = { algorithm: { category: 0 for _, category in categories.items() } for algorithm in ['knn', 'nb', 'rf'] }
    for _, results in dataset_results.items():
        for algorithm, category in results.items():
            summary[algorithm][category] += 1
    return summary

def extract_results_10x4cv(input_path, filtered_data_sets, pipeline, categories, folds, repeat):
    from sklearn.model_selection import RepeatedKFold

    # load and format the results
    simple_results = load_results(input_path, filtered_data_sets)
    simple_results = rich_simple_results(simple_results, pipeline, categories)

    # summarize the results
    grouped_by_algorithm_results, grouped_by_data_set_result = aggregate_results(simple_results, categories)
    
    rkf = RepeatedKFold(n_splits=folds, n_repeats=repeat, random_state=1)

    summaries = []
    datasets = list(grouped_by_data_set_result.keys())
    for train_index, test_index in rkf.split(datasets):
        train_dict = { datasets[your_key]: grouped_by_data_set_result[datasets[your_key]] for your_key in train_index }
        test_dict = { datasets[your_key]: grouped_by_data_set_result[datasets[your_key]] for your_key in test_index }
        train_per_algorithm = compute_summary_from_data_set_results(train_dict, categories)
        train_summary = compute_summary(train_per_algorithm, categories)
        train_per_algorithm['summary'] = train_summary
        test_per_algorithm = compute_summary_from_data_set_results(test_dict, categories)
        test_summary = compute_summary(test_per_algorithm, categories)
        test_per_algorithm['summary'] = test_summary
        summaries.append({'train': train_per_algorithm, 'test': test_per_algorithm})
    
    return summaries

def save_results(result_path, filtered_data_sets, simple_results, grouped_by_algorithm_results, summary):
    save_simple_results(result_path, simple_results, filtered_data_sets)
    save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, summary)

def save_results_10x4cv(result_path, summaries):
    for batch in range(len(summaries)):
        for set_, results in summaries[batch].items():
            with open(os.path.join(result_path, str(batch + 1) + set_ + '.csv'), 'w') as out:
                out.write(',' + ','.join(results['summary'].keys()) + '\n')
                for key, value in results.items():
                    row = key
                    for k, v in value.items():
                        row += ',' + str(v)
                    row += '\n'
                    out.write(row)
            


def merge_results(pipeline_results, algorithm_results):
    comparison = {}
    summary = {}
    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        summary[acronym] = {'algorithm': 0, 'pipeline': 0, 'draw': 0}
        comparison[acronym] = {}

    for key, value in pipeline_results.items():
        acronym = key.split('_')[0]
        data_set = key.split('_')[1]
        acc1 = pipeline_results[key]['conf1']['accuracy']
        acc2 = pipeline_results[key]['conf2']['accuracy']
        best_pipeline_result = pipeline_results[key]['conf' + str(1 if acc1 > acc2 else 2)]

        if algorithm_results[key]['baseline_score'] != best_pipeline_result['baseline_score']:
            print('Different baseline scores: ' + str(key) + ' ' + str(algorithm_results[key]['baseline_score']) + ' ' + str(best_pipeline_result['baseline_score']))

        comparison[acronym][data_set] = {'algorithm': algorithm_results[key]['accuracy'], 'pipeline': best_pipeline_result['accuracy'], 'baseline': algorithm_results[key]['baseline_score'], 'choosen_pipeline': best_pipeline_result['pipeline']}
        winner = 'algorithm' if comparison[acronym][data_set]['algorithm'] > comparison[acronym][data_set]['pipeline'] else ('pipeline' if comparison[acronym][data_set]['algorithm'] < comparison[acronym][data_set]['pipeline'] else 'draw')
        summary[acronym][winner] += 1

    new_summary = {'algorithm': 0, 'pipeline': 0, 'draw': 0}
    for algorithm, results in summary.items():
        for category, result in summary[algorithm].items():
            new_summary[category] += summary[algorithm][category]

    summary['summary'] = new_summary

    return comparison, summary

def save_comparison(comparison, result_path):
    def values_to_string(values):
        return [str(value).replace(',', '') for value in values]

    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        if os.path.exists('{}.csv'.format(acronym)):
            os.remove('{}.csv'.format(acronym))
        with open(os.path.join(result_path, '{}.csv'.format(acronym)), 'w') as out:
            keys = comparison[acronym][list(comparison[acronym].keys())[0]].keys()
            header = ','.join(keys)
            out.write('dataset,' + header + '\n')
            for dataset, results in comparison[acronym].items():
                result_string = ','.join(values_to_string(results.values()))
                out.write(dataset + ',' + result_string + '\n')

def save_summary(summary, result_path):
    if os.path.exists('summary.csv'):
        os.remove('summary.csv')
    with open(os.path.join(result_path, 'summary.csv'), 'w') as out:
        keys = summary[list(summary.keys())[0]].keys()
        header = ','.join(keys)
        out.write(',' + header + '\n')
        for algorithm, results in summary.items():
            result_string = ','.join([str(elem) for elem in results.values()])
            out.write(algorithm + ',' + result_string + '\n')






