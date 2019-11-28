import pandas as pd

def join_meta_features(impute):
    meta = pd.read_csv('../' + ('imputed-mean-' if impute else '') + 'extracted-meta-features-general.csv')
    meta2 = pd.read_csv('../' + ('imputed-mean-' if impute else '') + 'extracted-meta-features-model-landmarking.csv')

    join = pd.merge(meta, meta2, left_on='id', right_on='id')

    return join

def save_data_frame(result_path, data_frame, index):
    data_frame.to_csv(result_path, index=index)


for impute in [True, False]:
    data = join_meta_features(impute)
    save_data_frame('../' + ('imputed-mean-' if impute else '') + 'extracted-meta-features.csv', data, index=False)