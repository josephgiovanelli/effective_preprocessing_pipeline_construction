import collections
import os
import json

import pandas as pd

from os import listdir
from os.path import isfile, join

from commons import benchmark_suite, algorithms
from results_processors.correlation_utils import max_frequency, same_frequency


def create_possible_categories(pipeline):
    first = pipeline[0][0].upper()
    second = pipeline[1][0].upper()
    first_or_second = first + "o" + second
    first_second = first + second
    second_first = second + first
    draw = first_second + "o" + second_first
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
    df = pd.read_csv("../openml/meta-features.csv")
    df = df.loc[df['did'].isin(benchmark_suite)]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()

def merge_dict(list):
    """ Merge dictionaries and keep values of common keys in list"""
    new_dict = {}
    for key, value in list[0].items():
        new_value = []
        for dict in list:
            new_value.append(dict[key])
        new_dict[key] = new_value
    return new_dict

def load_results(input_path, filtered_datasets):
    comparison = {}
    confs = [os.path.join(input_path, "conf1"), os.path.join(input_path, "conf2")]
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
                        pipeline = str(data['context']['best_config']['pipeline']).replace(" ", "").replace(",", " ")
                        num_iterations = data['context']['iteration'] + 1
                        best_iteration = data['context']['best_config']['iteration'] + 1
                        baseline_score = data['context']['baseline_score'] // 0.0001 / 100
                else:
                    accuracy = 0
                    pipeline = ""
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


def save_simple_results(result_path, simple_results, filtered_datasets):
    def values_to_string(values):
        return [str(value).replace(',', '') for value in values]

    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        if os.path.exists('{}.csv'.format(acronym)):
            os.remove('{}.csv'.format(acronym))
        with open(os.path.join(result_path, '{}.csv'.format(acronym)), "w") as out:
            first_element = simple_results[list(simple_results.keys())[0]]
            conf_keys = first_element['conf1'].keys()
            conf1_header = ','.join([a + "1" for a in conf_keys])
            conf2_header = ','.join([a + "2" for a in conf_keys])
            result_header = ','.join(first_element['result'].keys())
            header = ','.join([result_header, conf1_header, conf2_header])
            out.write("dataset,name,dimensions," + header + '\n')


    df = pd.read_csv("../openml/meta-features.csv")
    df = df.loc[df['did'].isin(filtered_datasets)]

    for key, value in simple_results.items():
        acronym = key.split("_")[0]
        data_set = key.split("_")[1]
        name = df.loc[df['did'] == int(data_set)]['name'].values.tolist()[0]
        dimensions = ' x '.join([str(int(a)) for a in df.loc[df['did'] == int(data_set)][
            ['NumberOfInstances', 'NumberOfFeatures']].values.flatten().tolist()])


        with open(os.path.join(result_path, '{}.csv'.format(acronym)), "a") as out:
            results = ','.join(values_to_string(value['result'].values()))
            conf1 = ','.join(values_to_string(value['conf1'].values()))
            conf2 = ','.join(values_to_string(value['conf2'].values()))
            row = ','.join([data_set, name, dimensions, results, conf1, conf2])
            out.write(row + "\n")


def compose_pipeline(pipeline1, pipeline2, scheme):
    pipelines = {"pipeline1": [], "pipeline2": []}
    parameters = {"pipeline1": [], "pipeline2": []}
    for step in scheme:
        if pipeline1 != "":
            raw_pipeline1 = json.loads(pipeline1.replace('\'', '\"').replace(" ", ",").replace("True", "1").replace("False", "0"))
            pipelines["pipeline1"].append(raw_pipeline1[step][0].split("_")[1])
            for param in raw_pipeline1[step][1]:
                parameters["pipeline1"].append(raw_pipeline1[step][1][param])
        if pipeline2 != "":
            raw_pipeline2 = json.loads(pipeline2.replace('\'', '\"').replace(" ", ",").replace("True", "1").replace("False", "0"))
            pipelines["pipeline2"].append(raw_pipeline2[step][0].split("_")[1])
            for param in raw_pipeline2[step][1]:
                parameters["pipeline2"].append(raw_pipeline2[step][1][param])
    return pipelines, parameters


def check_validity(pipelines, result):
    if pipelines["pipeline1"] == [] and pipelines["pipeline2"] == []:
        validity, problem = False, 'not_exec'
    elif pipelines["pipeline1"] == [] or pipelines["pipeline2"] == []:
        validity, problem = False, 'not_exec_once'
    else:
        if pipelines["pipeline1"].__contains__('NoneType') and pipelines["pipeline2"].__contains__('NoneType'):
            validity = result == 0
        elif pipelines["pipeline1"].__contains__('NoneType') and not(pipelines["pipeline2"].__contains__('NoneType')):
            validity = result == 0 or result == 2
        elif not(pipelines["pipeline1"].__contains__('NoneType')) and pipelines["pipeline2"].__contains__('NoneType'):
            validity = result == 0 or result == 1
        else:
            validity = True
        problem = '' if validity else 'inconsistent'
    return validity, problem


def compute_result(result, pipelines, categories, baseline_scores, scores):
    if baseline_scores[0] != baseline_scores[1]:
        raise Exception('Baselines with different scores')

    #case a, b, c, e, i
    if result == 0 and baseline_scores[0] == scores[0]:
        return 'baseline'
    #case d, o
    elif pipelines["pipeline1"].count('NoneType') == 2 or pipelines["pipeline2"].count('NoneType') == 2:
        if pipelines["pipeline1"].count('NoneType') == 2:
            if result == 2:
                return categories['second_first']
            else:
                raise Exception("pipeline2 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores))
        else:
            if result == 1:
                return categories['first_second']
            else:
                raise Exception("pipeline1 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores))
    #case f, m, l, g
    elif pipelines["pipeline1"].count('NoneType') == 1 and pipelines["pipeline2"].count('NoneType') == 1:
        #case f
        if pipelines["pipeline1"][0] == 'NoneType' and pipelines["pipeline2"][0] == 'NoneType':
            if result == 0:
                return categories['second']
            else:
                raise Exception("pipelines is not drawing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores))
        #case m
        elif pipelines["pipeline1"][1] == 'NoneType' and pipelines["pipeline2"][1] == 'NoneType':
            if result == 0:
                return categories['first']
            else:
                raise Exception("pipelines is not drawing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm ")
        #case g, l
        elif (pipelines["pipeline1"][0] == 'NoneType' and pipelines["pipeline2"][1] == 'NoneType') or (pipelines["pipeline1"][1] == 'NoneType' and pipelines["pipeline2"][0] == 'NoneType'):
            if result == 0:
                return categories['first_or_second']
            else:
                raise Exception("pipelines is not drawing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm ")
    #case h, n
    elif pipelines["pipeline1"].count('NoneType') == 1:
        #case h
        if pipelines["pipeline1"][0] == 'NoneType':
            if result == 0:
                return categories['second']
            elif result == 2:
                return categories['second_first']
            else:
                raise Exception("pipeline2 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm ")
        #case n
        elif pipelines["pipeline1"][1] == 'NoneType':
            if result == 0:
                return categories['first']
            elif result == 2:
                return categories['second_first']
            else:
                raise Exception("pipeline2 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm ")
    # case p, q
    elif pipelines["pipeline2"].count('NoneType') == 1:
        # case p
        if pipelines["pipeline2"][0] == 'NoneType':
            if result == 0:
                return categories['second']
            elif result == 1:
                return categories['first_second']
            else:
                raise Exception("pipeline1 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm ")
        # case q
        elif pipelines["pipeline2"][1] == 'NoneType':
            if result == 0:
                return categories['second']
            elif result == 1:
                return categories['first_second']
            else:
                raise Exception("pipeline1 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm ")
    #case r
    elif pipelines["pipeline1"].count('NoneType') == 0 and pipelines["pipeline2"].count('NoneType') == 0:
        if result == 0:
            return categories['draw']
        elif result == 1:
            return categories['first_second']
        elif result == 2:
            return categories['second_first']
    else:
        raise Exception("This configuration matches nothing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm ")


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

        validity, label = check_validity(pipelines, winner)

        try:
            if validity:
                baseline_scores = [first_configuration['baseline_score'], second_configuration['baseline_score']]
                accuracies = [first_configuration['accuracy'], second_configuration['accuracy']]
                label = compute_result(winner, pipelines, categories, baseline_scores, accuracies)
        except Exception as e:
                print(str(e))

        first_configuration['pipeline'] = pipelines["pipeline1"]
        second_configuration['pipeline'] = pipelines["pipeline2"]

        first_configuration['parameters'] = parameters["pipeline1"]
        second_configuration['parameters'] = parameters["pipeline2"]

        value.append({'winner': winner, 'validity': validity, 'label': label})
        simple_results[key] = {'conf1': value[0], 'conf2': value[1], 'result': value[2]}
    return simple_results



def aggregate_results(simple_results, pipeline_scheme, categories):
    grouped_by_dataset_result = {}
    grouped_by_algorithm_results = {}

    for key, value in simple_results.items():
        acronym = key.split("_")[0]
        data_set = key.split("_")[1]
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
    with open(os.path.join(result_path, 'summary.csv'), "w") as out:
        out.write(',' + ','.join(summary.keys()) + '\n')
        if not(no_algorithms):
            for key, value in grouped_by_algorithm_results.items():
                row = key
                for k, v in value.items():
                    row += "," + str(v)
                row += "\n"
                out.write(row)
        row = "summary"
        for key, value in summary.items():
            row += "," + str(value)
        out.write(row)


def element_with_best_accuracy(results):
    indexed_results = []
    for r in results:
        indexed_results.append([r['accuracy'], r['result']])
    max = indexed_results[0][0]
    index = 0
    for i in range(1, len(indexed_results)):
        if indexed_results[i][0] > max:
            max = indexed_results[i][0]
            index = i
    return indexed_results[index][1]

def find_best_result(results):
    original_length = len(results)
    new_results = [r['result'] for r in results if r['result'] != 'inconsistent']
    if new_results == []:
        return 'inconsistent', 'inconsistent', original_length
    max_frequent_result, frequency = max_frequency(new_results)
    if same_frequency(new_results, max_frequent_result, frequency):
        results = [r for r in results if r['result'] != 'inconsistent']
        final_result = element_with_best_accuracy(results)
    else:
        final_result = max_frequent_result
    return final_result, max_frequent_result, frequency

def merge_runs_by_dataset(grouped_by_dataset_result, no_algorithms=False):

    details_grouped_by_dataset_result = merge_dict(grouped_by_dataset_result)
    new_grouped_by_dataset_result = {}


    for dataset, value in details_grouped_by_dataset_result.items():
        algorithms_dict = merge_dict(value)
        new_grouped_by_dataset_result[dataset] = {}
        details_grouped_by_dataset_result[dataset] = {}

        if no_algorithms:
            results = [item for sublist in list(algorithms_dict.values()) for item in sublist]
            final_result, max_frequent_result, frequency = find_best_result(results)
            details_grouped_by_dataset_result[dataset]["noalgorithm"] = {'results': [r['result'] for r in results], 'final_result': final_result,
                                              'max_frequent_result': max_frequent_result, 'frequency': frequency}
            new_grouped_by_dataset_result[dataset]['noalgorithm'] = final_result
        else:
            for algorithm, results in algorithms_dict.items():
                final_result, max_frequent_result, frequency = find_best_result(results)
                algorithms_dict[algorithm] = {'results': [r['result'] for r in results], 'final_result': final_result,
                                              'max_frequent_result': max_frequent_result, 'frequency': frequency}


                new_grouped_by_dataset_result[dataset][algorithm] = final_result

            details_grouped_by_dataset_result[dataset] = algorithms_dict

    return details_grouped_by_dataset_result, new_grouped_by_dataset_result

def save_details_grouped_by_dataset_result(result_path, details_grouped_by_dataset_result, no_algorithms = False):
    if no_algorithms:
        mylist = ['NOALGORITHM']
    else:
        mylist = algorithms

    for element in mylist:
        header = False
        acronym = ''.join([a for a in element if a.isupper()]).lower()

        with open(os.path.join(result_path, '{}.csv'.format(acronym)), "w") as out:

            for dataset, detail_results in details_grouped_by_dataset_result.items():
                if not(header):
                    out.write(',' + ','.join(detail_results[acronym].keys()) + '\n')
                    header = True
                results = ','.join(list(str(elem).replace(',', '')
                                                 .replace('[', '')
                                                 .replace(']', '')
                                                 .replace('\'', '') for elem in detail_results[acronym].values()))
                out.write(dataset + ',' + results + '\n')

def grouped_by_dataset_to_grouped_by_algorithm(grouped_by_dataset_result, categories, no_algorithms = False):
    grouped_by_algorithm_results = {}
    if no_algorithms:
        mylist = ['NOALGORITHM']
    else:
        mylist = algorithms

    for element in mylist:
        acronym = ''.join([a for a in element if a.isupper()]).lower()
        grouped_by_algorithm_results[acronym] = {}
        for _, category in categories.items():
            grouped_by_algorithm_results[acronym][category] = 0

    for _, result in grouped_by_dataset_result.items():
        for acronym, category in result.items():
            grouped_by_algorithm_results[acronym][category] += 1

    return grouped_by_algorithm_results





