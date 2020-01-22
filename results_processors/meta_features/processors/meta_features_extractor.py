import openml
import pandas as pd
from pymfe.mfe import MFE
from sklearn.impute import SimpleImputer


benchmark_suite = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307,
                       300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1510, 1489, 1494, 1497, 1501,
                       1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381, 40499,
                       40668, 40966, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978,
                       40670, 40701]

def get_filtered_datasets():
    df = pd.read_csv("../simple-meta-features.csv")
    df = df.loc[df['did'].isin(benchmark_suite)]
    df = df.loc[df['NumberOfMissingValues'] / (df['NumberOfInstances'] * df['NumberOfFeatures']) < 0.1]
    df = df.loc[df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances'] < 0.1]
    df = df.loc[df['NumberOfInstances'] * df['NumberOfFeatures'] < 5000000]
    df = df['did']
    return df.values.flatten().tolist()

for impute in [True, False]:
    for general in [True, False]:

        meta_features = []
        for id in get_filtered_datasets():
            dataset = openml.datasets.get_dataset(id)
            X, y, categorical_indicator, _ = dataset.get_data(
                dataset_format='array',
                target=dataset.default_target_attribute)

            if impute:
                X = SimpleImputer(strategy="constant").fit_transform(X)

            dict = {'id': id}
            print(id, dataset.name)

            if general:
                mfe = MFE(groups=["general", "statistical", "info-theory"])
            else:
                mfe = MFE(groups=["model-based", "landmarking"])

            mfe.fit(X, y)
            ft = mfe.extract()

            for i in range(0, len(ft[0])):
                dict[ft[0][i]] = ft[1][i]
            dict["nr_cat"] = len([i for i, x in enumerate(categorical_indicator) if x == True])
            dict["nr_num"] = len([i for i, x in enumerate(categorical_indicator) if x == False])

            meta_features.append(dict)

        df = pd.DataFrame(meta_features)

        name = ('imputed-' if impute else '') + 'extracted-meta-features' + ('-general' if general else 'model-landmarking')
        df.to_csv('../' + name + '.csv', index=False)
