import autosklearn.classification
import pandas as pd
import sklearn.model_selection
import sklearn.metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

import json

def encode_data(data):
    numeric_features = data.drop(columns = ['algorithm', 'class']).columns
    categorical_features = data[['algorithm', 'class']].columns
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

data = pd.read_csv('../../results/pipeline/features_rebalance/meta_learner/ts_all.csv')

columns = data.columns
data = SimpleImputer(strategy="constant", fill_value=0.0).fit_transform(data)
data = pd.DataFrame(data=data, columns=columns)
data = encode_data(data)

X, y = data.drop(columns = ['class']), data['class']

print(data)
estimator = autosklearn.classification.AutoSklearnClassifier()

# scores = cross_validate(estimator,
#                         X,
#                         encode_data(pd.DataFrame(y)),
#                         scoring=["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"],
#                         cv = 10,
#                         return_train_score=True,
#                         return_estimator=True)
# results = {'train': {
#     'accuracy': (scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2),
#     'balanced_accuracy': (scores['train_balanced_accuracy'].mean(), scores['train_balanced_accuracy'].std() * 2),
#     'precision_macro': (scores['train_precision_macro'].mean(), scores['train_precision_macro'].std() * 2),
#     'recall_macro': (scores['train_recall_macro'].mean(), scores['train_recall_macro'].std() * 2),
#     'f1_macro': (scores['train_f1_macro'].mean(), scores['train_f1_macro'].std() * 2),
# }, 'test': {
#     'accuracy': (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2),
#     'balanced_accuracy': (scores['test_balanced_accuracy'].mean(), scores['test_balanced_accuracy'].std() * 2),
#     'precision_macro': (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std() * 2),
#     'recall_macro': (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std() * 2),
#     'f1_macro': (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std() * 2),
# }}

y_hat = cross_val_predict(estimator, X, y, cv=88)
results = {'leave-one-out': {
    'accuracy': accuracy_score(y, y_hat),
    'balanced_accuracy': balanced_accuracy_score(y, y_hat).tolist(),
    'precision': precision_score(y, y_hat, average=None).tolist(),
    'recall': recall_score(y, y_hat, average=None).tolist(),
    'f1': f1_score(y, y_hat, average=None).tolist(),
    'confusion_matrix': confusion_matrix(y, y_hat, labels=["FR", "RF"]).tolist()
}}

print(results)

with open('auto-sklearn_two_classes_loo.json', 'w') as outfile:
    json.dump(results, outfile, indent=4)

