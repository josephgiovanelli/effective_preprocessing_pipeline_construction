from yellowbrick.target import FeatureCorrelation
from sklearn.preprocessing import OrdinalEncoder

from scipy import stats as s

import os

import numpy as np
import pandas as pd

from commons import algorithms


def max_frequency(x):
    counter = 0
    num = x[0]

    for i in x:
        curr_frequency = x.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num, counter

def same_frequency(x, num, frequency):
    for i in list(filter(lambda a: a != num, x)):
        curr_frequency = x.count(i)
        if curr_frequency == frequency:
            return True
    return False


def create_num_equal_elements_matrix(grouped_by_dataset_result):
    num_equal_elements_matrix = np.zeros((len(algorithms), len(algorithms)))

    for dataset, value in grouped_by_dataset_result.items():
        list_values = []
        for _, label in value.items():
            if label != 'inconsistent' and label != 'not_exec' and label != 'not_exec_once' and label != 'no_majority':
                list_values.append(label)
        if list_values:
            _, freq = max_frequency(list_values)
            num_equal_elements_matrix[len(list_values) - 1][freq - 1] += 1

    return num_equal_elements_matrix

def save_num_equal_elements_matrix(result_path, num_equal_elements_matrix):
    with open(os.path.join(result_path, 'num_equal_elements_matrix.csv'), "w") as out:
        out.write("length," + ",".join(str(i) for i in range(1, len(algorithms) + 1)) + ",tot\n")
        for i in range(0, np.size(num_equal_elements_matrix, 0)):
            row = str(i + 1)
            sum = 0
            for j in range(0, np.size(num_equal_elements_matrix, 1)):
                value = int(num_equal_elements_matrix[i][j])
                sum += value
                row += "," + str(value)
            row += "," + str(sum) + "\n"
            out.write(row)


def create_hamming_matrix(X, y):
    def hamming_distance(s1, s2):
        assert len(s1) == len(s2)
        return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))
    hamming_matrix = np.zeros((len(algorithms), len(algorithms)))
    value = np.zeros(len(algorithms))
    value[X[0][1]] = y[0] + 1
    for i in range(1, np.size(X, 0)):
        if X[i][0] == X[i-1][0]:
            value[X[i][1]] = y[i] + 1
        else:
            most_frequent = int(s.mode([x for x in value if x != 0])[0])
            weight = list(value).count(0)
            ideal = np.zeros(len(algorithms))
            for j in range(0, np.size(value, 0)):
                if value[j] != 0:
                    ideal[j] = most_frequent
            hamming_matrix[weight][hamming_distance(value, ideal)] += 1

            value = np.zeros(5)
            value[X[i][1]] = y[i] + 1
    return hamming_matrix



def create_correlation_matrix(filtered_datasets, grouped_by_dataset_result, categories, consider_just_the_order):
    data = []
    for dataset, value in grouped_by_dataset_result.items():
        for algorithm, result in value.items():
            if result != "inconsistent" and result != "not_exec" and result != "not_exec_once" and result != "no_majority":
                if consider_just_the_order:
                    data.append([dataset, algorithm, 1 if result == categories['first_second'] or result == categories['second_first'] else 0])
                else:
                    data.append([dataset, algorithm, result])

    df = pd.DataFrame(data)
    df.columns = ['dataset', 'algorithm', 'class']

    meta = pd.read_csv("../openml/meta-features.csv")
    meta = meta.loc[meta['did'].isin(filtered_datasets)]

    join = pd.merge(df.astype(str), meta.astype(str), left_on="dataset", right_on="did")
    join = join.drop(columns=["version", "status", "format", "uploader", "did", "row"])
    encoded = pd.DataFrame(OrdinalEncoder().fit_transform(join), columns=join.columns)

    kendall = encoded.corr(method ='kendall')['class'].to_frame()
    pearson = encoded.corr(method ='pearson')['class'].to_frame()
    spearman = encoded.corr(method ='spearman')['class'].to_frame()
    kendall.columns = ['kendall']
    pearson.columns = ['pearson']
    spearman.columns = ['spearman']

    correlation_matrix = pd.concat([kendall, pearson, spearman], axis=1, sort=False)

    X, y = encoded.drop(columns = ["class"]), encoded["class"]
    visualizer = FeatureCorrelation(method='mutual_info-classification', labels=X.columns)
    visualizer.fit(X, y)

    correlation_matrix = correlation_matrix.drop("class", axis = 0)
    correlation_matrix['mutual_info-classification'] = visualizer.scores_.tolist()

    return correlation_matrix

def save_correlation_matrix(result_path, correlation_matrix):
    with open(os.path.join(result_path, 'correlation_matrix.csv'), "w") as out:
        out.write(correlation_matrix.to_csv())