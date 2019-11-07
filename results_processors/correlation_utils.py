from scipy.stats import chi2_contingency, chi2
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from yellowbrick.target import FeatureCorrelation
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, FunctionTransformer

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
    with open(os.path.join(result_path, 'num_equal_elements_matrix.csv'), 'w') as out:
        out.write('length,' + ','.join(str(i) for i in range(1, len(algorithms) + 1)) + ',tot\n')
        for i in range(0, np.size(num_equal_elements_matrix, 0)):
            row = str(i + 1)
            sum = 0
            for j in range(0, np.size(num_equal_elements_matrix, 1)):
                value = int(num_equal_elements_matrix[i][j])
                sum += value
                row += ',' + str(value)
            row += ',' + str(sum) + '\n'
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
            if result != 'inconsistent' and result != 'not_exec' and result != 'no_majority':
                if consider_just_the_order:
                    data.append([int(dataset), algorithm, 'order' if result == categories['first_second'] or result == categories['second_first'] or result == 'not_exec_once' else 'no_order'])
                else:
                    data.append([int(dataset), algorithm, result])

    df = pd.DataFrame(data)
    df.columns = ['dataset', 'algorithm', 'class']

    meta = pd.read_csv('../openml/meta-features.csv')
    meta = meta.loc[meta['did'].isin(filtered_datasets)]
    meta = meta.drop(columns=['version', 'status', 'format', 'uploader', 'row', 'name'])
    meta = meta.astype(int)

    join = pd.merge(df, meta, left_on='dataset', right_on='did')
    join = join.drop(columns=['did'])
    with open(os.path.join('../results/features_rebalance/summary', 'join.csv'), 'w') as out:
        out.write(join.to_csv())
    numeric_features = join.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    categorical_features = join.select_dtypes(include=['object']).columns
    reorder_features = list(numeric_features) + list(categorical_features)
    encoded = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('a', FunctionTransformer())]), join.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns),
        ('cat', Pipeline(steps=[('b', OrdinalEncoder())]), join.select_dtypes(include=['object']).columns)]).fit_transform(join)
    encoded = pd.DataFrame(encoded, columns = reorder_features)
    with open(os.path.join('../results/features_rebalance/summary', 'encoded.csv'), 'w') as out:
        out.write(encoded.to_csv())

    kendall = encoded.corr(method ='kendall')['class'].to_frame()
    pearson = encoded.corr(method ='pearson')['class'].to_frame()
    spearman = encoded.corr(method ='spearman')['class'].to_frame()
    kendall.columns = ['kendall']
    pearson.columns = ['pearson']
    spearman.columns = ['spearman']

    correlation_matrix = pd.concat([kendall, pearson, spearman], axis=1, sort=False)

    X, y = encoded.drop(columns = ['class']), encoded['class']
    visualizer = FeatureCorrelation(method='mutual_info-classification', labels=X.columns)
    visualizer.fit(X, y)

    correlation_matrix = correlation_matrix.drop('class', axis = 0)
    correlation_matrix['mutual_info-classification'] = visualizer.scores_.tolist()

    return correlation_matrix

def save_correlation_matrix(result_path, correlation_matrix, consider_just_the_order):
    with open(os.path.join(result_path, 'correlation_matrix' + ( '' if not consider_just_the_order else '_order') + '.csv'), 'w') as out:
        out.write(correlation_matrix.to_csv())

def chi2test(observed, uniform_distribution):
    # comparing with the uniform distribution
    table = [observed, uniform_distribution]
    stat, p, dof, expected = chi2_contingency(table)
    # print(table)
    # print(expected)
    # print()

    # interpret test-statistic
    # stat is high as much as the two distribution are different
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    statistic_test = abs(stat) >= critical
    # print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    # if statistic_test:
    #     print('reject H0 -> No similarity with uniform distribution -> there is a majority -> one of them is more frequent')
    # else:
    #     print('fail to reject H0 -> similarity found -> there is NOT a majority -> balanced frequencies')
    # print()

    # interpret p-value
    # p is high as much as the two distribution are similar (the frequencies are balanced)
    alpha = 1.0 - prob
    p_value = p <= alpha
    # print('significance=%.3f, p=%.3f' % (alpha, p))
    # if p_value:
    #     print('reject H0 -> No similarity with uniform distribution -> there is a majority -> one of them is more frequent')
    # else:
    #     print('fail to reject H0 -> similarity found -> there is NOT a majority -> balanced frequencies')
    # print()
    return critical // 0.0001 / 10000, stat // 0.0001 / 10000, statistic_test, alpha // 0.0001 / 10000, p // 0.0001 / 10000, p_value

def chi2tests(grouped_by_algorithm_results, summary, categories):
    grouped_by_algorithm_results['summary'] = summary
    test = {}
    order_test = {}
    not_order_test = {}
    for algorithm, values in grouped_by_algorithm_results.items():

        # print('ALGORITHM: ' + algorithm)
        total = sum(a for a in values.values())
        not_valid = sum([values[categories['inconsistent']], values[categories['not_exec']], values[categories['not_exec_once']]])
        valid = total - not_valid

        order = sum([values[categories['first_second']], values[categories['second_first']]])
        not_order = valid - order
        uniform_frequency = valid / 2
        critical, stat, statistic_test, alpha, p, p_value = chi2test([order, not_order],
                                                                     [uniform_frequency, uniform_frequency])

        test[algorithm] = {'valid': valid,
                           'order': order,
                           'not_order': not_order,
                           'uniform_frequency': uniform_frequency,
                           'critical': critical,
                           'stat': stat,
                           'statistic_test': statistic_test,
                           'alpha': alpha,
                           'p': p,
                           'p_value_test': p_value}

        first_second = values[categories['first_second']]
        second_first = values[categories['second_first']]
        uniform_frequency = order / 2
        critical, stat, statistic_test, alpha, p, p_value = chi2test([first_second, second_first],
                                                                     [uniform_frequency, uniform_frequency])

        order_test[algorithm] = {'order': order,
                                 categories['first_second']: first_second,
                                 categories['second_first']: second_first,
                                 'uniform_frequency': uniform_frequency,
                                 'critical': critical,
                                 'stat': stat,
                                 'statistic_test': statistic_test,
                                 'alpha': alpha,
                                 'p': p,
                                 'p_value_test': p_value}

        first = values[categories['first']]
        second = values[categories['second']]
        first_or_second = values[categories['first_or_second']]
        draw = values[categories['draw']]
        baseline = values[categories['baseline']]
        uniform_frequency = not_order / 5
        critical, stat, statistic_test, alpha, p, p_value = chi2test([first, second, first_or_second, draw, baseline],
                                                                     [uniform_frequency, uniform_frequency, uniform_frequency, uniform_frequency, uniform_frequency])

        not_order_test[algorithm] = {'not_order': not_order,
                                     categories['first']: first,
                                     categories['second']: second,
                                     categories['first_or_second']: first_or_second,
                                     categories['draw']: draw,
                                     categories['baseline']: baseline,
                                     'uniform_frequency': uniform_frequency,
                                     'critical': critical,
                                     'stat': stat,
                                     'statistic_test': statistic_test,
                                     'alpha': alpha,
                                     'p': p,
                                     'p_value_test': p_value}

    return test, order_test, not_order_test


def save_chi2tests(result_path, test, order_test, not_order_test):
    def saver(collection, name):
        with open(name, 'w') as out:
            header = False
            for algorithm, values in collection.items():
                if not header:
                    out.write(',' + ','.join(values.keys()) + '\n')
                    header = True
                row = algorithm
                for _, value in values.items():
                    row += ',' + str(value)
                row += '\n'
                out.write(row)

    saver(test, os.path.join(result_path, 'test.csv'))
    saver(order_test, os.path.join(result_path, 'order_test.csv'))
    saver(not_order_test, os.path.join(result_path, 'not_order_test.csv'))




