import openml
import pandas as pd
from pymfe.mfe import MFE

from commons import benchmark_suite

def get_filtered_datasets():
    df = pd.read_csv("../openml/simple-meta-features.csv")
    df = df.loc[df['did'].isin(benchmark_suite)]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()


meta_features = []
for id in get_filtered_datasets():
    dataset = openml.datasets.get_dataset(id)
    X, y, _, _ = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute)
    dict = {'id': id}
    print(id, dataset.name)

    # Extract all measures
    mfe = MFE()
    mfe.fit(X, y)
    ft = mfe.extract()

    for i in range(0, len(ft[0])):
        dict[ft[0][i]] = ft[1][i]

    meta_features.append(dict)
    print(dict)

df = pd.DataFrame(meta_features)
df.to_csv('../openml/meta-features-extracted.csv', index=False)
