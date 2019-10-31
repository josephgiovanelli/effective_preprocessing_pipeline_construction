import collections
import os
import json

import pandas as pd

from os import listdir
from os.path import isfile, join

from commons import benchmark_suite, algorithms

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

def merge_dict(dict1, dict2):
    """ Merge dictionaries and keep values of common keys in list"""
    dict3 = {**dict2, **dict1}
    for key, value in dict3.items():
        if key in dict2 and key in dict1:
            dict3[key] = [value, dict2[key]]

    return dict3

def load_results(input_paths, filtered_datasets):
    comparison = {}
    for path in input_paths:
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
                comparison[path][acronym]['pipeline'] = pipeline
                comparison[path][acronym]['num_iterations'] = num_iterations
                comparison[path][acronym]['best_iteration'] = best_iteration
                comparison[path][acronym]['baseline_score'] = baseline_score

    return collections.OrderedDict(sorted(merge_dict(comparison[input_paths[0]], comparison[input_paths[1]]).items()))


def save_simple_results(result_path, simple_results, filtered_datasets):
    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        if os.path.exists('{}.csv'.format(acronym)):
            os.remove('{}.csv'.format(acronym))
        with open(os.path.join(result_path, '{}.csv'.format(acronym)), "a") as out:
            out.write("dataset,name,dimensions,conf1,pipeline1,num_iterations1,best_iteration1,conf2,pipeline2,"
                      "num_iterations2,best_iteration2\n")

    df = pd.read_csv("../openml/meta-features.csv")
    df = df.loc[df['did'].isin(filtered_datasets)]

    for key, value in simple_results.items():
        acronym = key.split("_")[0]
        dataset = key.split("_")[1]
        name = df.loc[df['did'] == int(dataset)]['name'].values.tolist()[0]
        dimensions = ' x '.join([str(int(a)) for a in df.loc[df['did'] == int(dataset)][
            ['NumberOfInstances', 'NumberOfFeatures']].values.flatten().tolist()])

        with open(os.path.join(result_path, '{}.csv'.format(acronym)), "a") as out:
            out.write(dataset + "," + name + "," + dimensions + "," + str(value[0]['accuracy']) + "," +
                      str(value[0]['pipeline']) + "," + str(value[0]['num_iterations']) + "," +
                      str(value[0]['best_iteration']) + ","+ str(value[0]['accuracy']) + "," +
                      str(value[1]['pipeline']) + "," + str(value[1]['num_iterations']) + "," +
                      str(value[1]['best_iteration']) + "\n")


def compose_pipeline(pipeline1, pipeline2, scheme):
    pipelines = {"pipeline1": [], "pipeline2": []}
    for step in scheme:
        if pipeline1 != "":
            raw_pipeline1 = json.loads(pipeline1.replace('\'', '\"').replace(" ", ",").replace("True", "1").replace("False", "0"))
            pipelines["pipeline1"].append(raw_pipeline1[step][0].split("_")[1])
        if pipeline2 != "":
            raw_pipeline2 = json.loads(pipeline2.replace('\'', '\"').replace(" ", ",").replace("True", "1").replace("False", "0"))
            pipelines["pipeline2"].append(raw_pipeline2[step][0].split("_")[1])
    return pipelines


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


def compute_result(result, dataset, acronym, grouped_by_algorithm_results, grouped_by_dataset_result, pipelines, categories, baseline_scores, scores):
    if baseline_scores[0] != baseline_scores[1]:
        print('Baselines with different scores')

    #case a, b, c, e, i
    if result == 0 and baseline_scores[0] == scores[0]:
        grouped_by_dataset_result[dataset][acronym] = 'baseline'
        grouped_by_algorithm_results[acronym]['baseline'] += 1
    #case d, o
    elif pipelines["pipeline1"].count('NoneType') == 2 or pipelines["pipeline2"].count('NoneType') == 2:
        if pipelines["pipeline1"].count('NoneType') == 2:
            if result == 2:
                grouped_by_dataset_result[dataset][acronym] = categories['second_first']
                grouped_by_algorithm_results[acronym][categories['second_first']] += 1
            else:
                print("pipeline2 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
        else:
            if result == 1:
                grouped_by_dataset_result[dataset][acronym] = categories['first_second']
                grouped_by_algorithm_results[acronym][categories['first_second']] += 1
            else:
                print("pipeline1 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
    #case f, m, l, g
    elif pipelines["pipeline1"].count('NoneType') == 1 and pipelines["pipeline2"].count('NoneType') == 1:
        #case f
        if pipelines["pipeline1"][0] == 'NoneType' and pipelines["pipeline2"][0] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = categories['second']
                grouped_by_algorithm_results[acronym][categories['second']] += 1
            else:
                print("pipelines is not drawing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
        #case m
        elif pipelines["pipeline1"][1] == 'NoneType' and pipelines["pipeline2"][1] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = categories['first']
                grouped_by_algorithm_results[acronym][categories['first']] += 1
            else:
                print("pipelines is not drawing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
        #case g, l
        elif (pipelines["pipeline1"][0] == 'NoneType' and pipelines["pipeline2"][1] == 'NoneType') or (pipelines["pipeline1"][1] == 'NoneType' and pipelines["pipeline2"][0] == 'NoneType'):
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = categories['first_or_second']
                grouped_by_algorithm_results[acronym][categories['first_or_second']] += 1
            else:
                print("pipelines is not drawing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
    #case h, n
    elif pipelines["pipeline1"].count('NoneType') == 1:
        #case h
        if pipelines["pipeline1"][0] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = categories['second']
                grouped_by_algorithm_results[acronym][categories['second']] += 1
            elif result == 2:
                grouped_by_dataset_result[dataset][acronym] = categories['second_first']
                grouped_by_algorithm_results[acronym][categories['second_first']] += 1
            else:
                print("pipeline2 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
        #case n
        elif pipelines["pipeline1"][1] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = categories['first']
                grouped_by_algorithm_results[acronym][categories['first']] += 1
            elif result == 2:
                grouped_by_dataset_result[dataset][acronym] = categories['second_first']
                grouped_by_algorithm_results[acronym][categories['second_first']] += 1
            else:
                print("pipeline2 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
    # case p, q
    elif pipelines["pipeline2"].count('NoneType') == 1:
        # case p
        if pipelines["pipeline2"][0] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = categories['second']
                grouped_by_algorithm_results[acronym][categories['second']] += 1
            elif result == 1:
                grouped_by_dataset_result[dataset][acronym] = categories['first_second']
                grouped_by_algorithm_results[acronym][categories['first_second']] += 1
            else:
                print("pipeline1 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
        # case q
        elif pipelines["pipeline2"][1] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = categories['second']
                grouped_by_algorithm_results[acronym][categories['second']] += 1
            elif result == 1:
                grouped_by_dataset_result[dataset][acronym] = categories['first_second']
                grouped_by_algorithm_results[acronym][categories['first_second']] += 1
            else:
                print("pipeline1 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
    #case r
    elif pipelines["pipeline1"].count('NoneType') == 0 and pipelines["pipeline2"].count('NoneType') == 0:
        if result == 0:
            grouped_by_dataset_result[dataset][acronym] = categories['draw']
            grouped_by_algorithm_results[acronym][categories['draw']] += 1
        elif result == 1:
            grouped_by_dataset_result[dataset][acronym] = categories['first_second']
            grouped_by_algorithm_results[acronym][categories['first_second']] += 1
        elif result == 2:
            grouped_by_dataset_result[dataset][acronym] = categories['second_first']
            grouped_by_algorithm_results[acronym][categories['second_first']] += 1
    else:
        print("This configuration matches nothing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))

    return grouped_by_algorithm_results, grouped_by_dataset_result

def aggregate_results(simple_results, pipeline, categories):
    grouped_by_dataset_result = {}
    grouped_by_algorithm_results = {}

    for key, value in simple_results.items():
        acronym = key.split("_")[0]
        dataset = key.split("_")[1]
        pipelines = compose_pipeline(value[0]['pipeline'], value[1]['pipeline'], pipeline)
        result = -1
        if value[0]['accuracy'] > value[1]['accuracy']:
            result = 1
        elif value[0]['accuracy'] == value[1]['accuracy']:
            result = 0
        elif value[0]['accuracy'] < value[1]['accuracy']:
            result = 2
        if result == -1:
            raise ValueError('A very bad thing happened.')

        validity, problem = check_validity(pipelines, result)

        if not(grouped_by_dataset_result.__contains__(dataset)):
            grouped_by_dataset_result[dataset] = {}

        if not(grouped_by_algorithm_results.__contains__(acronym)):
            grouped_by_algorithm_results[acronym] = {}
            for _, category in categories.items():
                grouped_by_algorithm_results[acronym][category] = 0

        if validity:
            grouped_by_algorithm_results, grouped_by_dataset_result = compute_result(
                result, dataset, acronym, grouped_by_algorithm_results, grouped_by_dataset_result, pipelines, categories,
                [value[0]['baseline_score'], value[1]['baseline_score']], [value[0]['accuracy'], value[1]['accuracy']])
        else:
            grouped_by_dataset_result[dataset][acronym] = problem
            grouped_by_algorithm_results[acronym][problem] += 1

    return grouped_by_algorithm_results, grouped_by_dataset_result

def save_grouped_by_algorithm_results(result_path, grouped_by_algorithm_results, categories):
    summary = {}
    for _, category in categories.items():
        summary[category] = sum(x[category] for x in grouped_by_algorithm_results.values())

    with open(os.path.join(result_path, 'grouped_by_algorithm_results.csv'), "w") as out:
        out.write(',' + ','.join(summary.keys()) + '\n')
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