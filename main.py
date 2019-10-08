from experiment.pipeline.PrototypeSingleton import PrototypeSingleton
from experiment.utils import scenarios, serializer, cli, datasets
from experiment import policies

import json

import openml

from sklearn.model_selection import train_test_split


def main(args):
    scenario = scenarios.load(args.scenario)
    scenario = cli.apply_scenario_customization(scenario, args.customize)
    config = scenarios.to_config(scenario)
    print('SCENARIO:\n {}'.format(json.dumps(scenario, indent=4, sort_keys=True)))

    task = openml.tasks.get_task(scenario['setup']['dataset'])
    X, y = task.get_X_and_y()
    print(X, y)

    PrototypeSingleton.getInstance().setPipeline(args.pipeline)
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


if __name__ == "__main__":
    args = cli.parse_args()
    main(args)