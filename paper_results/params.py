

PARAMS={
    'maritime': {'lr': 1e-4,
                 'bs': 16,
                 'h': 512,
                 'n_layers': 1,
                 'd': 0.1,
                 'init_crel': 2
                 },
    # UCR 
    # 'ECG200': {'lr': 1e-5,
    #              'bs': 32,
    #              'h': 512,
    #              'n_layers': 0,
    #              'd': 0.2,
    #              'init_crel': 1
    #              },
    # 'ECG5000': {'lr': 1e-5,
    #              'bs': 32,
    #              'h': 512,
    #              'n_layers': 0,
    #              'd': 0.2,
    #              'init_crel': 1
    #              },
    'EOGVerticalSignal': {'lr': 1e-4,
                 'bs': 64,
                 'h': 128,
                 'n_layers': 1,
                 'd': 0.2,
                 'init_crel': 2
                 },
    'Epilepsy2': {'lr': 1e-4,
                 'bs': 16,
                 'h': 128,
                 'n_layers': 0,
                 'd': 0.1,
                 'init_crel': 6
                 },
    'GunPoint': {'lr': 1e-4,
                 'bs': 64,
                 'h': 512,
                 'n_layers': 0,
                 'd': 0.2,
                 'init_crel': 1
                 },
    'GunPointOldVersusYoung': {'lr': 1e-4,
                 'bs': 32,
                 'h': 128,
                 'n_layers': 0,
                 'd': 0.2,
                 'init_crel': 5
                 },
    'NerveDamage': {'lr': 2e-5,
                 'bs': 32,
                 'h': 256,
                 'n_layers': 0,
                 'd': 0.1,
                 'init_crel': 1
                 },
    'SharePriceIncrease': {'lr': 2e-6,
                 'bs': 32,
                 'h': 256,
                 'n_layers': 2,
                 'd': 0.2,
                 'init_crel': 4
                 },
    # UEA
    'HandMovementDirection': {'lr': 1e-5,
                 'bs': 32,
                 'h': 512,
                 'n_layers': 0,
                 'd': 0.2,
                 'init_crel': 1
                 },
}

def get_params(dataset):
    return PARAMS[dataset]