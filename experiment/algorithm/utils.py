from hyperopt import hp

def generate_domain_space(prototype):
    domain_space = {}
    for k, v in prototype.items():
        domain_space[k] = hp.choice(k, v)
    return domain_space
