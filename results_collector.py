from __future__ import print_function
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, SVR
from yellowbrick.target import FeatureCorrelation
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import OrdinalEncoder
from scipy import stats as s


import argparse
import collections
import os
import json

import pandas as pd
import numpy as np

algorithms = ['RandomForest', 'NaiveBayes', 'KNearestNeighbors', 'SVM', 'NeuralNet']
benchmark_suite = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307,
                       300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1510, 1489, 1494, 1497, 1501,
                       1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381, 40499,
                       40668, 40966, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978,
                       40670, 40701]

def mergeDict(dict1, dict2):
    """ Merge dictionaries and keep values of common keys in list"""
    dict3 = {**dict2, **dict1}
    for key, value in dict3.items():
        if key in dict2 and key in dict1:
            dict3[key] = [value, dict2[key]]

    return dict3

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Machine Learning Workflow creation and configuration")
    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=True, help="step of the pipeline to execute")
    parser.add_argument("-i", "--first_input", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-ii", "--second_input", nargs="?", type=str, required=True, help="path of second input")
    parser.add_argument("-o", "--output", nargs="?", type=str, required=True, help="path where put the results")
    args = parser.parse_args()
    input_paths = [args.first_input, args.second_input]
    result_path = args.output
    pipeline = args.pipeline
    return input_paths, result_path, pipeline

def get_filtered_datasets():
    df = pd.read_csv("openml/meta-features.csv")
    df = df.loc[df['did'].isin(benchmark_suite)]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()

def load_results(input_paths, filtered_datasets, comparison):
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
                        num_iterations = data['context']['iteration']
                        best_iteration = data['context']['best_config']['iteration']
                        baseline_score = data['context']['baseline_score'] // 0.0001 / 100
                        comparison[path][acronym] = (accuracy, pipeline, num_iterations, best_iteration, baseline_score)
                else:
                    accuracy = 0
                    pipeline = ""
                    num_iterations = 0
                    best_iteration = 0
                    baseline_score = 0
                    comparison[path][acronym] = (accuracy, pipeline, num_iterations, best_iteration, baseline_score)

    return collections.OrderedDict(sorted(mergeDict(comparison[input_paths[0]], comparison[input_paths[1]]).items()))

def create_output_files(result_path, grouped_by_algorithm_results, scheme):
    first = scheme[0][0].upper()
    second = scheme[1][0].upper()
    firstOrSecond = first + "o" + second
    firstSecond = first + second
    secondFirst = second + first
    draws = firstSecond + "o" + secondFirst

    for algorithm in algorithms:
        acronym = ''.join([a for a in algorithm if a.isupper()]).lower()
        grouped_by_algorithm_results[acronym] = {firstSecond: 0, draws: 0, secondFirst: 0, 'NN': 0, first: 0, second: 0, firstOrSecond: 0, 'valid': 0}
        with open(os.path.join(result_path, '{}.csv'.format(acronym)), "a") as out:
            out.write("dataset,name,dimensions,conf1,pipeline1,num_iterations1,best_iteration1,conf2,pipeline2,"
                      "num_iterations2,best_iteration2,valid\n")
    return grouped_by_algorithm_results

def compose_pipeline(pipeline1, pipeline2, scheme):
    pipelines = {"pipeline1": [], "pipeline2": []}
    for step in scheme:
        if pipeline1 != "":
            raw_pipeline1 = json.loads(pipeline1.replace('\'', '\"').replace(" ", ","))
            pipelines["pipeline1"].append(raw_pipeline1[step][0].split("_")[1])
        if pipeline2 != "":
            raw_pipeline2 = json.loads(pipeline2.replace('\'', '\"').replace(" ", ","))
            pipelines["pipeline2"].append(raw_pipeline2[step][0].split("_")[1])
    return pipelines


def check_validity(pipelines, result):
    if pipelines["pipeline1"] == [] or pipelines["pipeline1"] == []:
        return False
    else:
        if pipelines["pipeline1"].__contains__('NoneType') and pipelines["pipeline2"].__contains__('NoneType'):
            return result == 0
        if pipelines["pipeline1"].__contains__('NoneType') and not(pipelines["pipeline2"].__contains__('NoneType')):
            return result == 0 or result == 2
        if not(pipelines["pipeline1"].__contains__('NoneType')) and pipelines["pipeline2"].__contains__('NoneType'):
            return result == 0 or result == 1
    return True


def compute_result(result, dataset, acronym, grouped_by_algorithm_results, grouped_by_dataset_result, pipelines, scheme, baseline_scores, scores):
    if baseline_scores[0] != baseline_scores[1]:
        print('Baselines with different scores')

    first = scheme[0][0].upper()
    second = scheme[1][0].upper()
    firstOrSecond = first + "o" + second
    firstSecond = first + second
    secondFirst = second + first
    draws = firstSecond + "o" + secondFirst

    #case a, b, c, e, i
    if result == 0 and baseline_scores[0] == scores[0]:
        grouped_by_dataset_result[dataset][acronym] = 'NN'
        grouped_by_algorithm_results[acronym]['NN'] += 1
    #case d, o
    elif pipelines["pipeline1"].count('NoneType') == 2 or pipelines["pipeline2"].count('NoneType') == 2:
        if pipelines["pipeline1"].count('NoneType') == 2:
            if result == 2:
                grouped_by_dataset_result[dataset][acronym] = secondFirst
                grouped_by_algorithm_results[acronym][secondFirst] += 1
            else:
                print("pipeline2 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
        else:
            if result == 1:
                grouped_by_dataset_result[dataset][acronym] = firstSecond
                grouped_by_algorithm_results[acronym][firstSecond] += 1
            else:
                print("pipeline1 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
    #case f, m, l, g
    elif pipelines["pipeline1"].count('NoneType') == 1 and pipelines["pipeline2"].count('NoneType') == 1:
        #case f
        if pipelines["pipeline1"][0] == 'NoneType' and pipelines["pipeline2"][0] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = second
                grouped_by_algorithm_results[acronym][second] += 1
            else:
                print("pipelines is not drawing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
        #case m
        elif pipelines["pipeline1"][1] == 'NoneType' and pipelines["pipeline2"][1] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = first
                grouped_by_algorithm_results[acronym][first] += 1
            else:
                print("pipelines is not drawing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
        #case g, l
        elif (pipelines["pipeline1"][0] == 'NoneType' and pipelines["pipeline2"][1] == 'NoneType') or (pipelines["pipeline1"][1] == 'NoneType' and pipelines["pipeline2"][0] == 'NoneType'):
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = firstOrSecond
                grouped_by_algorithm_results[acronym][firstOrSecond] += 1
            else:
                print("pipelines is not drawing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
    #case h, n
    elif pipelines["pipeline1"].count('NoneType') == 1:
        #case h
        if pipelines["pipeline1"][0] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = second
                grouped_by_algorithm_results[acronym][second] += 1
            elif result == 2:
                grouped_by_dataset_result[dataset][acronym] = secondFirst
                grouped_by_algorithm_results[acronym][secondFirst] += 1
            else:
                print("pipeline2 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
        #case n
        elif pipelines["pipeline1"][1] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = first
                grouped_by_algorithm_results[acronym][first] += 1
            elif result == 2:
                grouped_by_dataset_result[dataset][acronym] = secondFirst
                grouped_by_algorithm_results[acronym][secondFirst] += 1
            else:
                print("pipeline2 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
    # case p, q
    elif pipelines["pipeline2"].count('NoneType') == 1:
        # case p
        if pipelines["pipeline2"][0] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = second
                grouped_by_algorithm_results[acronym][second] += 1
            elif result == 1:
                grouped_by_dataset_result[dataset][acronym] = firstSecond
                grouped_by_algorithm_results[acronym][firstSecond] += 1
            else:
                print("pipeline1 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
        # case q
        elif pipelines["pipeline2"][1] == 'NoneType':
            if result == 0:
                grouped_by_dataset_result[dataset][acronym] = second
                grouped_by_algorithm_results[acronym][second] += 1
            elif result == 1:
                grouped_by_dataset_result[dataset][acronym] = firstSecond
                grouped_by_algorithm_results[acronym][firstSecond] += 1
            else:
                print("pipeline1 is not winning. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))
    #case r
    elif pipelines["pipeline1"].count('NoneType') == 0 and pipelines["pipeline2"].count('NoneType') == 0:
        if result == 0:
            grouped_by_dataset_result[dataset][acronym] = draws
            grouped_by_algorithm_results[acronym][draws] += 1
        elif result == 1:
            grouped_by_dataset_result[dataset][acronym] = firstSecond
            grouped_by_algorithm_results[acronym][firstSecond] += 1
        elif result == 2:
            grouped_by_dataset_result[dataset][acronym] = secondFirst
            grouped_by_algorithm_results[acronym][secondFirst] += 1
    else:
        print("This configuration matches nothing. " + str(pipelines) + " baseline_score " + str(baseline_scores[0]) + " scores " + str(scores) + " algorithm " + str(acronym))

    return grouped_by_algorithm_results, grouped_by_dataset_result


def aggregate_results(filtered_datasets, results, grouped_by_algorithm_results, result_path, pipeline):
    df = pd.read_csv("openml/meta-features.csv")
    df = df.loc[df['did'].isin(filtered_datasets)]
    first = pipeline[0][0].upper()
    second = pipeline[1][0].upper()
    firstOrSecond = first + "o" + second
    firstSecond = first + second
    secondFirst = second + first
    draws = firstSecond + "o" + secondFirst

    firstOrSecond = first + "o" + second
    grouped_by_dataset_result = {}
    for key, value in results.items():
        acronym = key.split("_")[0]
        dataset = key.split("_")[1]
        name = df.loc[df['did'] == int(dataset)]['name'].values.tolist()[0]
        dimensions = ' x '.join([str(int(a)) for a in df.loc[df['did'] == int(dataset)][['NumberOfInstances', 'NumberOfFeatures']].values.flatten().tolist()])
        conf1 = value[0][0]
        conf2 = value[1][0]
        pipelines = compose_pipeline(value[0][1], value[1][1], pipeline)
        result = -1
        if conf1 > conf2:
            result = 1
        elif conf1 == conf2:
            result = 0
        elif conf1 < conf2:
            result = 2
        if result == -1:
            raise ValueError('A very bad thing happened.')
        if acronym == "knn" and result == 0 and abs(conf1 - conf2) < 1:
            result = 0
        valid = check_validity(pipelines, result)
        if valid:
            if not(grouped_by_dataset_result.__contains__(dataset)):
                grouped_by_dataset_result[dataset] = {}
            grouped_by_algorithm_results, grouped_by_dataset_result = compute_result(
                result, dataset, acronym, grouped_by_algorithm_results, grouped_by_dataset_result, pipelines, pipeline,
                [value[0][4], value[1][4]], [conf1, conf2])
            grouped_by_algorithm_results[acronym]['valid'] += 1

        with open(os.path.join(result_path, '{}.csv'.format(acronym)), "a") as out:
            out.write(dataset + "," + name + "," + dimensions + "," + str(conf1) + "," + str(value[0][1]) + "," +
                      str(value[0][2]) + "," + str(value[0][3]) + "," + str(conf2) + "," + str(value[1][1]) +
                      "," + str(value[1][2]) +  "," + str(value[1][3]) +  "," + str(valid) + "\n")

    summary = {firstSecond: sum(x[firstSecond] for x in grouped_by_algorithm_results.values()),
                     draws: sum(x[draws] for x in grouped_by_algorithm_results.values()),
                     secondFirst: sum(x[secondFirst] for x in grouped_by_algorithm_results.values()),
                     'NN': sum(x['NN'] for x in grouped_by_algorithm_results.values()),
                     first: sum(x[first] for x in grouped_by_algorithm_results.values()),
                     second: sum(x[second] for x in grouped_by_algorithm_results.values()),
                     firstOrSecond: sum(x[firstOrSecond] for x in grouped_by_algorithm_results.values()),
                     'valid': sum(x['valid'] for x in grouped_by_algorithm_results.values())}
    return grouped_by_algorithm_results, grouped_by_dataset_result, summary

def write_summary(result_path, grouped_by_algorithm_results, summary, scheme):
    first = scheme[0][0].upper()
    second = scheme[1][0].upper()
    firstOrSecond = first + "o" + second
    firstSecond = first + second
    secondFirst = second + first
    draws = firstSecond + "o" + secondFirst
    with open(os.path.join(result_path, 'results.csv'), "a") as out:
        out.write("algorithm," + firstSecond + "," + draws + "," + secondFirst + ",NN," + first + "," + second + "," + firstOrSecond + ",valid\n")
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

def max_frequency(list):
    counter = 0
    num = list[0]

    for i in list:
        curr_frequency = list.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num, counter



def create_num_equal_elements_matrix(result_path, grouped_by_dataset_result):
    num_equal_elements = np.zeros((5, 5))

    for dataset, value in grouped_by_dataset_result.items():
        list_values = []
        for k, v in value.items():
            list_values.append(v)
        _, freq = max_frequency(list_values)
        num_equal_elements[len(value) - 1][freq - 1] += 1

    with open(os.path.join(result_path, 'num_equal_elements_matrix.csv'), "a") as out:
        out.write("length,1,2,3,4,5,tot\n")
        for i in range(0, np.size(num_equal_elements, 0)):
            row = str(i + 1)
            for j in range(0, np.size(num_equal_elements, 1)):
                row += "," + str(int(num_equal_elements[i][j]))
            row += "," + str(len(list(filter(lambda y: len(y) == i + 1, grouped_by_dataset_result.values()))))
            row += "\n"
            out.write(row)


def create_hamming_matrix(result_path, X, y):
    def hamming_distance(s1, s2):
        assert len(s1) == len(s2)
        return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))
    hamming_matrix = np.zeros((5, 5))
    value = np.zeros(5)
    value[X[0][1]] = y[0] + 1
    for i in range(1, np.size(X, 0)):
        if X[i][0] == X[i-1][0]:
            value[X[i][1]] = y[i] + 1
        else:
            most_frequent = int(s.mode([x for x in value if x != 0])[0])
            weight = list(value).count(0)
            ideal = np.zeros(5)
            for j in range(0, np.size(value, 0)):
                if value[j] != 0:
                    ideal[j] = most_frequent
            hamming_matrix[weight][hamming_distance(value, ideal)] += 1

            value = np.zeros(5)
            value[X[i][1]] = y[i] + 1
    print(hamming_matrix)



def create_correlation_matrix(result_path, grouped_by_dataset_result):
    data = []
    for dataset, value in grouped_by_dataset_result.items():
        for algorithm, result in value.items():
            data.append([dataset, algorithm, result])

    df = pd.DataFrame(OrdinalEncoder().fit_transform(data))
    kendall = df.corr(method='kendall')
    pearson = df.corr(method='pearson')
    spearman = df.corr(method='spearman')

    f = np.array(['dataset', 'algorithm'])
    data = OrdinalEncoder().fit_transform(data)
    X = data[:,:-1]
    y = data[:,-1]
    create_hamming_matrix(result_path, X.astype(int), y.astype(int))
    visualizer = FeatureCorrelation(method='mutual_info-regression', labels=f)
    visualizer.fit(X, y, discrete_features=[True, True], random_state=0)
    mutual_info_regression = visualizer.scores_

    estimator = SVR(kernel="linear")
    selector = RFE(estimator, step=1, n_features_to_select=2)
    selector = selector.fit(X, y)
    rfe_coefficient = selector.estimator_.coef_

    with open(os.path.join(result_path, 'correlation_matrix.csv'), "a") as out:
        out.write(",kendall,pearson,spearman,mutual_info_regression,rfe_coefficient\n")
        out.write("dataset," + str(kendall[0][2]) + "," + str(pearson[0][2]) + "," + str(spearman[0][2]) + "," +
                  str(mutual_info_regression[0]) + "," + str(rfe_coefficient[0][0]) + "\n")
        out.write("algorithm," + str(kendall[1][2]) + "," + str(pearson[1][2]) + "," + str(spearman[1][2]) + "," +
                  str(mutual_info_regression[1]) + "," + str(rfe_coefficient[0][1]))

def main():
    input_paths, result_path, pipeline = parse_args()
    filtered_datasets = get_filtered_datasets()
    results = load_results(input_paths, filtered_datasets, {})
    grouped_by_algorithm_results = create_output_files(result_path, {}, pipeline)
    grouped_by_algorithm_results, grouped_by_dataset_result, summary = aggregate_results(
        filtered_datasets, results, grouped_by_algorithm_results, result_path, pipeline)
    write_summary(result_path, grouped_by_algorithm_results, summary, pipeline)
    create_num_equal_elements_matrix(result_path, grouped_by_dataset_result)
    create_correlation_matrix(result_path, grouped_by_dataset_result)

main()









