from __future__ import print_function

import argparse
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

algorithms = ['RandomForest', 'DecisionTree', 'SVM', 'NeuralNet']

comparison = {}

for path in input_paths:
    files = [f for f in listdir(path) if isfile(join(path, f))]
    results = [f for f in files if f[-4:] == 'json']

    comparison[path] = {}
    for result in results:
        with open(os.path.join(path, result)) as json_file:
            data = json.load(json_file)
            comparison[path][result[:-5]] = data['context']['max_history_score']

results = mergeDict(comparison[input_paths[0]], comparison[input_paths[1]])
partial_results = {}

for algorithm in algorithms:
    acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
    partial_results[acronym] = {'conf1': 0, 'draws': 0, 'conf2': 0}
    with open(os.path.join(result_path, '{}.csv'.format(acronym)), "a") as out:
        out.write("dataset,accuracy1,accuracy2\n")

for key, value in results.items():
    acronym = key.split("_")[0]
    dataset = key.split("_")[1]
    partial_results[acronym]['conf1'] += 1 if value[0] - value[1] >= 0.001 else 0
    partial_results[acronym]['draws'] += 1 if (value[0] - value[1] <= 0.001) and (value[1] - value[0] <= 0.001) else 0
    partial_results[acronym]['conf2'] += 1 if value[1] - value[0] >= 0.001 else 0
    with open(os.path.join(result_path, '{}.csv'.format(acronym)), "a") as out:
        out.write(dataset + "," + str(value[0]) + "," + str(value[1]) + "\n")

complete_results = {'conf1': sum(value[0] - value[1] >= 0.001 for value in results.values()),
                 'draws': sum((value[0] - value[1] <= 0.001) and (value[1] - value[0] <= 0.001) for value in results.values()),
                 'conf2': sum(value[1] - value[0] >= 0.001 for value in results.values())}

with open(os.path.join(result_path, 'results.txt'), "a") as out:
    out.write("GROUPED BY THE ALGORITHMS")
    for key, value in partial_results.items():
        out.write("\n\n" + key + "\n")
        for k, v in value.items():
            out.write(k + ": " + str(v) + "     ")
    out.write("\n\n\n\nSUMMARY\n\n")
    for key, value in complete_results.items():
        out.write(key + ": " + str(value) + "     ")




