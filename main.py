from sklearn.impute import SimpleImputer

from experiment.pipeline.PrototypeSingleton import PrototypeSingleton
from experiment.utils import scenarios, serializer, cli, datasets
from experiment import policies

import json

import openml

from sklearn.model_selection import train_test_split

def load_dataset(id):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )
    print(dataset.name)
    print(X, y)
    num_features = [i for i, x in enumerate(categorical_indicator) if x == False]
    cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
    print("numeriche: " + str(len(num_features)) + " categoriche: " + str(len(cat_features)))
    PrototypeSingleton.getInstance().setFeatures(num_features, cat_features)
    PrototypeSingleton.getInstance().set_X_y(X, y)
    return X, y

def main(args):
    scenario = scenarios.load(args.scenario)
    scenario = cli.apply_scenario_customization(scenario, args.customize)
    config = scenarios.to_config(scenario)

    PrototypeSingleton.getInstance().setPipeline(getPipeline(scenario['setup']['dataset'], config['algorithm']))
    print(getPipeline(scenario['setup']['dataset'], config['algorithm']))

    print('SCENARIO:\n {}'.format(json.dumps(scenario, indent=4, sort_keys=True)))

    X, y = load_dataset(scenario['setup']['dataset'])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=scenario['control']['seed']
    )

    policy = policies.initiate(scenario['setup']['policy'], config)
    policy.run(X, y)

    serializer.serialize_results(scenario, policy, args.result_path)

def getPipeline(id, algorithm):
    if algorithm == 'KNearestNeighbors':
        if id == 3:
            return ['impute', 'encode', 'normalize', 'rebalance', 'features']
        if id == 6:
            return ['impute', 'encode', 'discretize', 'rebalance', 'features']
        if id == 16:
            return ['impute', 'encode', 'rebalance', 'discretize', 'features']
        if id == 182:
            return ['impute', 'encode', 'normalize', 'rebalance', 'features']
        if id == 300:
            return ['impute', 'encode', 'normalize', 'features', 'rebalance']
        if id == 469:
            return ['impute', 'encode', 'discretize', 'features', 'rebalance']
        if id == 1461:
            return ['impute', 'encode', 'normalize', 'rebalance', 'features']
        if id == 1468:
            return ['impute', 'encode', 'normalize', 'features', 'rebalance']
        if id == 1494:
            return ['impute', 'encode', 'normalize', 'features', 'rebalance']
        if id == 40979:
            return ['impute', 'encode', 'normalize', 'features', 'rebalance']

    if algorithm == 'NaiveBayes':
        if id == 3:
            return ['impute', 'encode', 'normalize', 'features', 'rebalance']
        if id == 6:
            return ['impute', 'encode', 'normalize', 'features', 'rebalance']
        if id == 16:
            return ['impute', 'encode', 'normalize', 'features', 'rebalance']
        if id == 182:
            return ['impute', 'encode', 'discretize', 'features', 'rebalance']
        if id == 300:
            return ['impute', 'encode', 'normalize', 'features', 'rebalance']
        if id == 469:
            return ['impute', 'encode', 'discretize', 'rebalance', 'features']
        if id == 1461:
            return ['impute', 'encode', 'normalize', 'features', 'rebalance']
        if id == 1468:
            return ['impute', 'encode', 'rebalance', 'discretize', 'features']
        if id == 1494:
            return ['impute', 'encode', 'discretize', 'features', 'rebalance']
        if id == 40979:
            return ['impute', 'encode', 'normalize', 'features', 'rebalance']

    if algorithm == 'RandomForest':
        if id == 3:
            return ['impute', 'encode', 'rebalance', 'discretize', 'features']
        if id == 6:
            return ['impute', 'encode', 'normalize', 'rebalance', 'features']
        if id == 16:
            return ['impute', 'encode', 'discretize', 'rebalance', 'features']
        if id == 182:
            return ['impute', 'encode', 'normalize', 'rebalance', 'features']
        if id == 300:
            return ['impute', 'encode', 'discretize', 'rebalance', 'features']
        if id == 469:
            return ['impute', 'encode', 'rebalance', 'discretize', 'features']
        if id == 1461:
            return ['impute', 'encode', 'normalize', 'rebalance', 'features']
        if id == 1468:
            return ['impute', 'encode', 'rebalance', 'discretize', 'features']
        if id == 1494:
            return ['impute', 'encode', 'discretize', 'features', 'rebalance']
        if id == 40979:
            return ['impute', 'encode', 'discretize', 'rebalance', 'features']


if __name__ == "__main__":
    args = cli.parse_args()
    main(args)