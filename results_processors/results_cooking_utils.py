from scipy.stats import chi2_contingency, chi2
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from yellowbrick.target import FeatureCorrelation
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, FunctionTransformer

from scipy import stats as s

import os

import numpy as np
import pandas as pd

algorithms = ['RandomForest', 'NaiveBayes', 'KNearestNeighbors']


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

def get_results(grouped_by_dataset_result):
    data = []
    for dataset, value in grouped_by_dataset_result.items():
        for algorithm, result in value.items():
            if result != 'inconsistent' and result != 'not_exec' and result != 'no_majority':
                data.append([int(dataset), algorithm, result])

    df = pd.DataFrame(data)
    df.columns = ['dataset', 'algorithm', 'class']
    return df

def encode_data(data):
    numeric_features = data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    categorical_features = data.select_dtypes(include=['object']).columns
    reorder_features = list(numeric_features) + list(categorical_features)
    encoded = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('a', FunctionTransformer())]),
             numeric_features),
            ('cat', Pipeline(steps=[('b', OrdinalEncoder())]),
             categorical_features)
        ]).fit_transform(data)
    encoded = pd.DataFrame(encoded, columns=reorder_features)

    return encoded

def join_result_with_extended_meta_features(filtered_datasets, data):
    meta = pd.read_csv('meta_features/extended-meta-features.csv')
    meta = meta.loc[meta['id'].isin(filtered_datasets)]
    meta = meta.drop(columns=['name', 'runs'])

    join = pd.merge(data, meta, left_on='dataset', right_on='id')
    join = join.drop(columns=['id'])

    return join

def join_result_with_extracted_meta_features(data, impute):
    meta = pd.read_csv('meta_features/' + ('imputed-mean-' if impute else '') + 'extracted-meta-features.csv', index_col=False)

    join = pd.merge(meta, data, left_on='id', right_on='dataset')
    join = join.drop(columns=['id'])

    return join


def join_result_with_simple_meta_features(filtered_datasets, data):
    meta = pd.read_csv('results_processors/meta_features/simple-meta-features.csv')
    meta = meta.loc[meta['did'].isin(filtered_datasets)]
    meta = meta.drop(columns=['version', 'status', 'format', 'uploader', 'row', 'name'])
    meta = meta.astype(int)

    join = pd.merge(data, meta, left_on='dataset', right_on='did')
    join = join.drop(columns=['did'])

    return join

def modify_class(data, categories, option):
    for key, value in categories.items():
        if option == 'group_all':
            if key == 'first_second' or key == 'second_first' or key == 'not_exec_once':
                data = data.replace(value, 'order')
            else:
                data = data.replace(value, 'no_order')
        if option == 'group_no_order':
            if key != 'first_second' and key != 'second_first' and key != 'not_exec_once':
                data = data.replace(value, 'no_order')
            if key == 'not_exec_once':
                data = data.drop(data.index[data['class'] == value].tolist())

    return data

def create_correlation_matrix(data):
    encoded = encode_data(data)

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

def save_data_frame(result_path, data_frame, index):
    data_frame.to_csv(result_path, index=index)

def save_correlation_matrix(result_path, name, correlation_matrix, group_no_order):
    save_data_frame(os.path.join(result_path, name + ( '_grouped' if group_no_order else '') + '.csv'), correlation_matrix, index=True)

def save_train_meta_learner(result_path, name, train_meta_learner, group_no_order):
    save_data_frame(os.path.join(result_path, name + ( '_grouped' if group_no_order else '') + '.csv'), train_meta_learner, index=False)

def chi2test(observed, distribution, prob = 0.95):
    # the print after the first '->' are valid just if we comparing the observed frequencies with the uniform distribution
    table = [observed, distribution]
    stat, p, dof, expected = chi2_contingency(table)
    # print(table)
    # print(expected)
    # print()

    # interpret test-statistic
    # stat is high as much as the two distribution are different
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

def chi2tests(grouped_by_algorithm_results, summary, categories, uniform):
    def run_chi2test(observed, uniform, formatted_input):
        tot = sum(observed)
        observed.sort(reverse=True)

        if uniform:
            length = len(observed)
            uniform_frequency = tot / length
            distribution = [uniform_frequency] * length
        else:
            distribution = [tot * 0.9, tot * 0.1]

        critical, stat, statistic_test, alpha, p, p_value = chi2test(observed, distribution)

        formatted_output = {'critical': critical,
                            'stat': stat,
                            'statistic_test': statistic_test,
                            'alpha': alpha,
                            'p': p,
                            'p_value_test': p_value}

        formatted_input.update(formatted_output)
        return formatted_input

    grouped_by_algorithm_results['summary'] = summary
    test = {}
    order_test = {}
    not_order_test = {}
    for algorithm, values in grouped_by_algorithm_results.items():

        total = sum(a for a in values.values())
        not_valid = sum([values[categories['inconsistent']], values[categories['not_exec']], values[categories['not_exec_once']]])
        valid = total - not_valid

        order = sum([values[categories['first_second']], values[categories['second_first']]])
        not_order = valid - order

        formatted_input = {'valid': valid,
                           'order': order,
                           'not_order': not_order}


        test[algorithm] = run_chi2test([order, not_order], uniform, formatted_input)


        first_second = values[categories['first_second']]
        second_first = values[categories['second_first']]

        formatted_input = {'order': order,
                                 categories['first_second']: first_second,
                                 categories['second_first']: second_first}

        order_test[algorithm] = run_chi2test([first_second, second_first], uniform, formatted_input)


        if uniform:
            first = values[categories['first']]
            second = values[categories['second']]
            first_or_second = values[categories['first_or_second']]
            draw = values[categories['draw']]
            baseline = values[categories['baseline']]

            formatted_input = {'not_order': not_order,
                                         categories['first']: first,
                                         categories['second']: second,
                                         categories['first_or_second']: first_or_second,
                                         categories['draw']: draw,
                                         categories['baseline']: baseline}

            not_order_test[algorithm] = run_chi2test([first_second, second_first], uniform, formatted_input)

    return {'test': test, 'order_test': order_test, 'not_order_test': not_order_test}


def save_chi2tests(result_path, tests):
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

    for key, value in tests.items():
        if value:
            saver(value, os.path.join(result_path, key + '.csv'))



