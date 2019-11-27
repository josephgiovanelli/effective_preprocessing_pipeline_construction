import pandas as pd
import sklearn.metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from results_processors.results_cooking_utils import encode_data


validation = ['train', 'test']

data = pd.read_csv('../../results/features_rebalance2/meta_learner/train_data_rf_grouped.csv')

#columns = data.columns
#data = SimpleImputer(strategy="constant").fit_transform(data)
#data = pd.DataFrame(data=data, columns=columns)

#data = encode_data(data)

X, y = data.drop(columns = ['class', 'algorithm', 'dataset']), data['class']

if validation == 'test':
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

ml =  RandomForestClassifier()

if validation == 'test':
    ml.fit(X_train, y_train)
else:
    ml.fit(X, y)

y_hat = ml.predict(X)
print("Accuracy score " + validation, sklearn.metrics.accuracy_score(y, y_hat))