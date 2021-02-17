import numpy as np
import openml
import pandas as pd
from pymfe.mfe import MFE
from sklearn.impute import SimpleImputer


benchmark_suite = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307,
                       300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1510, 1489, 1494, 1497, 1501,
                       1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381, 40499,
                       40668, 40966, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978,
                       40670, 40701]
extended_benchmark_suite = [41145, 41156, 41157, 4541, 41158, 42742, 40498, 42734, 41162, 42733, 42732, 1596, 40981, 40685, 
                        4135, 41142, 41161, 41159, 41163, 41164, 41138, 41143, 41146, 41150, 40900, 41165, 41166, 41168, 41169, 
                        41147, 1111, 1169, 41167, 41144, 1515, 1457, 181]

decode = False

def get_filtered_datasets():
    df = pd.read_csv("../simple-meta-features.csv")
    df = df.loc[df['did'].isin(extended_benchmark_suite)]
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

            if decode:
                if True in categorical_indicator:
                    X = pd.DataFrame(X)
                    for i in range(0, len(categorical_indicator)):
                        if categorical_indicator[i]:
                            X.iloc[:, i] = X.iloc[:, i].fillna(-1)
                            X.iloc[:, i] = X.iloc[:, i].astype('str')
                            X.iloc[:, i] = X.iloc[:, i].replace('-1', np.nan)
                            print("str")
                            print(X.iloc[:, i])
                            print()
                        else:
                            X.iloc[:, i] = X.iloc[:, i].fillna(-1)
                            X.iloc[:, i] = X.iloc[:, i].astype('float')
                            X.iloc[:, i] = X.iloc[:, i].replace(-1.0, np.nan)
                            print("float")
                            print(X.iloc[:, i])
                            print()


                    print(X)
                    X = X.values.tolist()
                    y = y.astype('str').tolist()


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

            meta_features.append(dict)

        df = pd.DataFrame(meta_features)

        name = ('imputed-' if impute else '') + 'extracted-meta-features' + ('-general' if general else '-model-landmarking')
        df.to_csv('../' + name + '.csv', index=False)
