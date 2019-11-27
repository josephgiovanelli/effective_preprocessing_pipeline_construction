import autosklearn.classification
import pandas as pd
import sklearn.model_selection
import sklearn.metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from results_processors.results_cooking_utils import encode_data

data = pd.read_csv('../../results/features_rebalance2/meta_learner/train_data_rf_grouped.csv')

#columns = data.columns
#data = SimpleImputer(strategy="constant").fit_transform(data)
#data = pd.DataFrame(data=data, columns=columns)

#data = encode_data(data)

X, y = data.drop(columns = ['class', 'algorithm', 'dataset']), data['class']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.classification.AutoSklearnClassifier()

automl.fit(X_train, y_train)

y_hat = automl.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))