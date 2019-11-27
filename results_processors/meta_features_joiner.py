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

def join_meta_features():
    meta = pd.read_csv('../openml/extracted-meta-features1.csv')
    meta2 = pd.read_csv('../openml/extracted-meta-features2.csv')

    join = pd.merge(meta2, meta, left_on='id', right_on='id')

    return join

def save_data_frame(result_path, data_frame, index):
    data_frame.to_csv(result_path, index=index)



data = join_meta_features()
save_data_frame('../openml/extracted-meta-features.csv', data, index=False)