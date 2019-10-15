from __future__ import print_function

import argparse
import collections
import os
from os import listdir
from os.path import isfile, join
import json

def mergeDict(dict1, dict2):
    """ Merge dictionaries and keep values of common keys in list"""
    dict3 = {**dict2, **dict1}
    for key, value in dict3.items():
        if key in dict2 and key in dict1:
            dict3[key] = [value, dict2[key]]

    return dict3

parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
parser.add_argument("-i", "--first_input", nargs="?", type=str, required=True, help="path of second input")
parser.add_argument("-ii", "--second_input", nargs="?", type=str, required=True, help="path of second input")
parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
args = parser.parse_args()

input_paths = [args.first_input, args.second_input]
result_path = args.output

algorithms = ['RandomForest', 'NaiveBayes', 'KNearestNeighbors', 'SVM', 'NeuralNet']

comparison = {}

for path in input_paths:
    files = [f for f in listdir(path) if isfile(join(path, f))]
    results = [f for f in files if f[-4:] == 'json']

    comparison[path] = {}
    for result in results:
        with open(os.path.join(path, result)) as json_file:
            data = json.load(json_file)
            accuracy = data['context']['max_history_score']
            pipeline = str(data['context']['best_config']['pipeline']).replace(",", " ")
            num_iterations = data['context']['iteration']
            best_iteration = data['context']['best_config']['iteration']
            start_time_best_iteration = data['context']['best_config']['start_time']
            stop_time_best_iteration = data['context']['best_config']['stop_time']
            comparison[path][result[:-5]] = (accuracy, pipeline, num_iterations, best_iteration, start_time_best_iteration, stop_time_best_iteration)

results = collections.OrderedDict(sorted(mergeDict(comparison[input_paths[0]], comparison[input_paths[1]]).items()))
partial_results = {}

for algorithm in algorithms:
    acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
    partial_results[acronym] = {'conf1': 0, 'draws': 0, 'conf2': 0}
    with open(os.path.join(result_path, '{}.csv'.format(acronym)), "a") as out:
        out.write("dataset,conf1,pipeline1,num_iterations1,best_iteration1,start_time_best_iteration1,"
                  "stop_time_best_iteration1,conf2,pipeline2,num_iterations2,best_iteration2,"
                  "start_time_best_iteration2,stop_time_best_iteration2\n")

pairs = []

for key, value in results.items():
    if len(value) == 2:
        pairs += [value]
        acronym = key.split("_")[0]
        dataset = key.split("_")[1]
        conf1 = value[0][0]
        conf2 = value[1][0]
        partial_results[acronym]['conf1'] += 1 if conf1 - conf2 >= 0.001 else 0
        partial_results[acronym]['draws'] += 1 if (conf1 - conf2 <= 0.001) and (conf2 - conf1 <= 0.001) else 0
        partial_results[acronym]['conf2'] += 1 if conf2 - conf1 >= 0.001 else 0
        with open(os.path.join(result_path, '{}.csv'.format(acronym)), "a") as out:
            out.write(dataset + "," + str(conf1) + "," + str(value[0][1]) + "," + str(value[0][2]) + "," +
                      str(value[0][3]) + "," + str(value[0][4]) + "," + str(value[0][5]) + "," + str(conf2) + "," +
                      str(value[1][1]) + "," + str(value[1][2]) + "," + str(value[1][3]) + "," + str(value[1][4]) +
                      "," + str(value[1][5]) + "\n")

complete_results = {'conf1': sum(value[0][0] - value[1][0] >= 0.001 for value in pairs),
                 'draws': sum((value[0][0] - value[1][0] <= 0.001) and (value[1][0] - value[0][0] <= 0.001) for value in pairs),
                 'conf2': sum(value[1][0] - value[0][0] >= 0.001 for value in pairs)}

with open(os.path.join(result_path, 'results.csv'), "a") as out:
    out.write("algorithm,conf1,draws,conf2\n")
    for key, value in partial_results.items():
        row = key
        for k, v in value.items():
            row += "," + str(v)
        row += "\n"
        out.write(row)
    row = "summary"
    for key, value in complete_results.items():
        row += "," + str(value)
    out.write(row)





