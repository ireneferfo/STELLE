

PARAMS={
    'maritime': {'lr': 1e-4,
                 'bs': 16,
                 'h': 512,
                 'n_layers': 1,
                 'd': 0.1,
                 'init_crel': 2
                 }
}

def get_params(dataset):
    return PARAMS[dataset]