from datetime import datetime
import json
import os

def serialize_results(scenario, policy, result_path):
    results = {
        'scenario': scenario,
        'context': policy.context
    }
    path = os.path.join(result_path, '{}.json'.format(scenario['file_name']))
    with open(path, 'w') as outfile:
        json.dump(results, outfile, indent=4)