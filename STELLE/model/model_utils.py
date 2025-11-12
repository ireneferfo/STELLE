import torch
import torch.optim as optim
from time import time
import os
from .model import ConceptBasedModel, ModelConfig


def train_test_model(args, type: str = 'base', **kwargs):
    """Train or load a concept-based model with the given configuration.
    Type (str): base, Robs, RobsAsHx, RobsAsGx, NoGx, Anchor
    """
    (
        kernel, trainloader, valloader, testloader, model_path_ev, config
    ) = args
    
    device = kernel.device
    checkpath = model_path_ev[:-3] + "_checkpoint.pt"
    
    # Try loading existing model
    loaded = _load_model(model_path_ev, device)
    if loaded:
        print(f'Loaded existing model from {model_path_ev}.')
        model, bestepoch, traintime, testresults, testtime = loaded
    else:
        # Create and train new model
        model = _create_model(kernel, trainloader, config, type, device, **kwargs)
        model, bestepoch, traintime = _train_model(
            model, trainloader, valloader, config.lr, config.epochs, config.verbose,
            checkpath, config.patience, config.cf, config.val_every_n_epochs
        )
        testresults, testtime = _validate(model, testloader)
        _save_model(model, bestepoch, traintime, testresults, testtime, model_path_ev, checkpath)
        model = model.to(device)
    
    results = {
        "best_epoch": bestepoch,
        **testresults,
        "train_time": traintime,
        "test_time": testtime
    }
    
    return model, results


def _create_model(kernel, trainloader, config, type, device, **kwargs):
    """Create a new ConceptBasedModel instance."""
    if type == 'Anchor':
        from .model_variants import AnchorModelConfig
        modelconfig = AnchorModelConfig(
        kernel=kernel,
        base_concepts = kernel.phis, 
        concepts=kwargs.get('concepts'),
        concept_embeddings = kwargs.get('concept_embeddings'), 
        num_classes=trainloader.dataset.num_classes,
        initial_epsilon=config.init_eps,
        initial_concept_relevance_scale=config.init_crel,
        dropout=config.d,
        parallel_workers=config.pll,
        hidden_size=config.h,
        num_layers=config.n_layers,
        )
    else:
        modelconfig = ModelConfig(
        kernel=kernel,
        concepts=kernel.phis,
        num_classes=trainloader.dataset.num_classes,
        initial_epsilon=config.init_eps,
        initial_concept_relevance_scale=config.init_crel,
        dropout=config.d,
        parallel_workers=config.pll,
        hidden_size=config.h,
        num_layers=config.n_layers,
        )
    match type:
        case 'base': 
            return ConceptBasedModel(modelconfig).to(device)
        case 'RobsAsGx':
            from .model_variants import ConceptBasedModel_RobsAsGx
            return ConceptBasedModel_RobsAsGx(modelconfig).to(device)
        case 'NoGx':
            from .model_variants import ConceptBasedModel_NoGx
            return ConceptBasedModel_NoGx(modelconfig).to(device)
        case 'RobsAsHx':
            from .model_variants import ConceptBasedModel_RobsAsHx
            return ConceptBasedModel_RobsAsHx(modelconfig).to(device)
        case 'Robs':
            from .model_variants import ConceptBasedModel_Robs
            return ConceptBasedModel_Robs(modelconfig).to(device)
        case 'Anchor':
            from .model_variants import ConceptBasedModel_Anchor
            return ConceptBasedModel_Anchor(modelconfig).to(device)
        case _:
            raise ValueError(f'Model of type {type} does not exist yet. Options: base, Robs, RobsAsHx, RobsAsGx, NoGx, Anchor')



def _train_model(model, trainloader, valloader, lr, epochs, verbose, checkpath, patience, cf, val_every_n_epochs):
    """Train the model with the specified configuration."""
    optimizer = _create_optimizer(model, lr)
    
    print("\nStarting model training...\n")
    _, bestepoch, traintime = model.train_model(
        optimizer,
        trainloader,
        valloader=valloader,
        num_epochs=int(epochs),
        verbose=verbose,
        checkpath=checkpath,
        patience=patience,
        check_frequency=cf,
        val_every_n_epochs=val_every_n_epochs,
    )
    
    print(f"\nBest epoch: {bestepoch}, took {traintime:.0f}s. Testing the model...")
    return model, bestepoch, traintime


def _create_optimizer(model, lr):
    """Create optimizer with separate parameter groups for regularization."""
    param_groups = [
        # Main parameters
        {
            "params": [
                p for n, p in model.named_parameters()
                if not n.endswith("_strength")
            ]
        },
        # Regularization controllers (10x higher LR)
        {
            "params": [
                p for n, p in model.named_parameters()
                if n.endswith("_strength")
            ],
            "lr": lr * 10,
        },
    ]
    return optim.Adam(param_groups, lr=lr)


def _validate(model, loader):
    """Run validation on the given data loader."""
    start_time = time()
    results = model.validate(loader, by_class_stats=False)
    total_time = time() - start_time
    
    print(
        f"Acc: {results['accuracy']:.1f} Wacc: {results['weighted_acc']:.1f}, "
        f"took {total_time:.0f}s."
    )
    return results, total_time


def _load_model(path, device):
    """Load an existing model from disk if it exists."""
    if not os.path.exists(path):
        return None
    
    try:
        print(f"\nLoading existing model from {path}.")
        model, bestepoch, traintime, testresults, testtime = torch.load(
            path, weights_only=False
        )
        model = model.to(device)
        model.device = device
        return model, bestepoch, traintime, testresults, testtime
    except Exception as e:
        print(f"Failed to load existing model ({e}), will train a new one.")
        return None


def _save_model(model, bestepoch, traintime, testresults, testtime, model_path, checkpath):
    """Save the trained model to disk and clean up checkpoint."""
    if os.path.exists(model_path):
        os.remove(model_path)
    
    torch.save(
        (model.cpu(), bestepoch, traintime, testresults, testtime),
        model_path
    )
    print(f"Trained model saved to {model_path}")
    
    # Remove checkpoint file
    if os.path.exists(checkpath):
        os.remove(checkpath)