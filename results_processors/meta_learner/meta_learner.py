import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_predict, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

import json

def encode_and_impute_data(data):
    numeric_features = data.drop(columns = ['algorithm']).columns
    categorical_features = data[['algorithm']].columns
    reorder_features = list(numeric_features) + list(categorical_features)
    encoded = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('a', SimpleImputer(strategy="mean"))]),
             numeric_features),
            ('cat', Pipeline(steps=[('b', OrdinalEncoder())]),
             categorical_features)
        ]).fit_transform(data)
    encoded = pd.DataFrame(encoded, columns=reorder_features)

    return encoded


def save_data_frame(result_path, data_frame, index):
    data_frame.to_csv(result_path, index=index)

validations = ['cross-validation', 'leave-one-out']
input_path = '../../results/pipeline/features_rebalance/meta_learner/training_set/'
output_path = '../../results/pipeline/features_rebalance/meta_learner/results/'
data_sets = ['three_classes/ts_rf', 'three_classes/ts_all', 'two_classes/ts_rf', 'two_classes/ts_all',]

for data_set in data_sets:

    data = pd.read_csv(input_path + data_set + '.csv')

    X, y = data.drop(columns = ['class']), data['class']

    if data_set.endswith('all'):
        X = encode_and_impute_data(X)
    else:
        columns = X.columns
        X = SimpleImputer(strategy="mean").fit_transform(X)
        X = pd.DataFrame(data=X, columns=columns)

    results = {}
    for validation in validations:

        results[validation] = {}

        estimator =  RandomForestClassifier()

        if validation == 'just-train':
            estimator.fit(X, y)
            y_hat = estimator.predict(X)
            results[validation] = {
                'accuracy': accuracy_score(y, y_hat),
                'balanced_accuracy': balanced_accuracy_score(y, y_hat),
                'precision': precision_score(y, y_hat, average=None).tolist(),
                'recall': recall_score(y, y_hat, average=None).tolist(),
                'f1': f1_score(y, y_hat, average=None).tolist(),
                'confusion_matrix': confusion_matrix(y, y_hat, labels=["no_order", "RF", "FR"]).tolist()
            }
        elif validation == 'hold-out':
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
            estimator.fit(X_train, y_train)
            y_hat = estimator.predict(X_train)
            results[validation]['train'] = {
                'accuracy': accuracy_score(y_train, y_hat),
                'balanced_accuracy': balanced_accuracy_score(y_train, y_hat),
                'precision': precision_score(y_train, y_hat, average=None).tolist(),
                'recall': recall_score(y_train, y_hat, average=None).tolist(),
                'f1': f1_score(y_train, y_hat, average=None).tolist(),
                'confusion_matrix': confusion_matrix(y_train, y_hat, labels=["no_order", "RF", "FR"]).tolist()
            }
            y_hat = estimator.predict(X_test)
            results[validation]['test'] = {
                'accuracy': accuracy_score(y_test, y_hat),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_hat),
                'precision': precision_score(y_test, y_hat, average=None).tolist(),
                'recall': recall_score(y_test, y_hat, average=None).tolist(),
                'f1': f1_score(y_test, y_hat, average=None).tolist(),
                'confusion_matrix': confusion_matrix(y_test, y_hat, labels=["no_order", "RF", "FR"]).tolist()
            }
        elif validation == 'cross-validation2':
            scores = cross_validate(estimator,
                                    X,
                                    y,
                                    scoring=["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"],
                                    cv = 10,
                                    return_train_score=True,
                                    return_estimator=True)
            results[validation]['train'] = {
                'accuracy': (scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2),
                'balanced_accuracy': (scores['train_balanced_accuracy'].mean(), scores['train_balanced_accuracy'].std() * 2),
                'precision_macro': (scores['train_precision_macro'].mean(), scores['train_precision_macro'].std() * 2),
                'recall_macro': (scores['train_recall_macro'].mean(), scores['train_recall_macro'].std() * 2),
                'f1_macro': (scores['train_f1_macro'].mean(), scores['train_f1_macro'].std() * 2),
            }
            results[validation]['test'] = {
                'accuracy': (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2),
                'balanced_accuracy': (scores['test_balanced_accuracy'].mean(), scores['test_balanced_accuracy'].std() * 2),
                'precision_macro': (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2),
                'recall_macro': (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2),
                'f1_macro': (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std() * 2),
            }
        elif validation == 'cross-validation':
            y_hat = cross_val_predict(estimator, X, y, cv=10)
            results[validation] = {
                'accuracy': accuracy_score(y, y_hat),
                'balanced_accuracy': balanced_accuracy_score(y, y_hat).tolist(),
                'precision': precision_score(y, y_hat, average=None).tolist(),
                'recall': recall_score(y, y_hat, average=None).tolist(),
                'f1': f1_score(y, y_hat, average=None).tolist(),
                'confusion_matrix': confusion_matrix(y, y_hat, labels=["no_order", "RF", "FR"]).tolist()
            }
        elif validation == 'leave-one-out':
            del results[validation]
            # results[validation + '1'] = {}
            # results[validation + '2'] = {}
            results[validation] = {}

            loo = LeaveOneOut()
            scores = {
                'test_accuracy': [],
                'test_balanced_accuracy': [],
                'test_precision_macro': [],
                'test_recall_macro': [],
                'test_f1_macro': [],
                'train_accuracy': [],
                'train_balanced_accuracy': [],
                'train_precision_macro': [],
                'train_recall_macro': [],
                'train_f1_macro': [],
            }
            y_hats = []
            X, y = X.values, y.values
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                estimator.fit(X_train, y_train)
                # y_hat = estimator.predict(X_train)
                # scores['train_accuracy'].append(accuracy_score(y_train, y_hat))
                # scores['train_balanced_accuracy'].append(balanced_accuracy_score(y_train, y_hat))
                # scores['train_precision_macro'].append(precision_score(y_train, y_hat, average='macro').tolist())
                # scores['train_recall_macro'].append(recall_score(y_train, y_hat, average='macro').tolist())
                # scores['train_f1_macro'].append(f1_score(y_train, y_hat, average='macro').tolist())
                y_hat = estimator.predict(X_test)
                y_hats.append(y_hat[0])
                # scores['test_accuracy'].append(accuracy_score(y_test, y_hat))
                # scores['test_balanced_accuracy'].append(balanced_accuracy_score(y_test, y_hat))
                # scores['test_precision_macro'].append(precision_score(y_test, y_hat, average='macro').tolist())
                # scores['test_recall_macro'].append(recall_score(y_test, y_hat, average='macro').tolist())
                # scores['test_f1_macro'].append(f1_score(y_test, y_hat, average='macro').tolist())

            for key, value in scores.items():
                scores[key] = np.asarray(value)

            # results[validation + '1']['train'] = {
            #     'accuracy': (scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2),
            #     'balanced_accuracy': (
            #     scores['train_balanced_accuracy'].mean(), scores['train_balanced_accuracy'].std() * 2),
            #     'precision_macro': (scores['train_precision_macro'].mean(), scores['train_precision_macro'].std() * 2),
            #     'recall_macro': (scores['train_recall_macro'].mean(), scores['train_recall_macro'].std() * 2),
            #     'f1_macro': (scores['train_f1_macro'].mean(), scores['train_f1_macro'].std() * 2),
            # }
            # results[validation + '1']['test'] = {
            #     'accuracy': (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2),
            #     'balanced_accuracy': (
            #     scores['test_balanced_accuracy'].mean(), scores['test_balanced_accuracy'].std() * 2),
            #     'precision_macro': (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2),
            #     'recall_macro': (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2),
            #     'f1_macro': (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std() * 2),
            # }
            results[validation] = {
                'accuracy': accuracy_score(y, y_hats),
                'balanced_accuracy': balanced_accuracy_score(y, y_hats),
                'precision': precision_score(y, y_hats, average=None).tolist(),
                'recall': recall_score(y, y_hats, average=None).tolist(),
                'f1': f1_score(y, y_hats, average=None).tolist(),
                'confusion_matrix': confusion_matrix(y, y_hats, labels=["no_order", "RF", "FR"]).tolist()
            }

    print(results)
    with open(output_path + data_set.replace('ts', 'results') + '.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)
