import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from time import time
import argparse
import numpy as np

import gc
import sys
import os
import csv
import pickle
import tempfile
import warnings
from itertools import product
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split


# ensure workspace root is on sys.path so the local `STELLE` package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from STELLE.data.dataset_utils import (
    remove_redundant_variables,
    convert_labels_to_numeric,
)
from STELLE.utils import get_device, set_all_possible_seeds, flatten_dict
from STELLE.data.base_dataset import TrajectoryDataset
from STELLE.kernels.base_measure import BaseMeasure
from STELLE.kernels.stl_kernel import StlKernel
from STELLE.kernels.trajectory_kernel import TrajectoryKernel
from STELLE.formula_generation.stl_generator import STLFormulaGenerator
from STELLE.formula_generation.formula_manager import FormulaManager
from STELLE.model.model import ConceptBasedModel, ModelConfig
from STELLE.explanations.global_explanation import get_training_explanations
from STELLE.explanations.explanation_metrics import get_local_metrics, get_global_metrics


# region FIXED PARAMETERS
n_train = 100
n_test = 80
n_vars = 2
series_length = 30
num_classes = 3

# n_trials = 40
SEED = 0
pll = 2
workers = 0  # os.cpu_count()
samples = 500
epochs = 5
cf = 50  # check frequency
patience = 5
val_every_n_epochs = 1
verbose = 10
logging = False
normalize = False
exp_kernel = False
normalize_rhotau = True
exp_rhotau = True
# concepts FIXED
# dim_concepts = dc = 100 # per var
t = 1 # 0.99
nvars_formulae = 1
creation_mode = "one"
dim_concepts = 10
min_total = 100  # ignore the minimum
imp_t_l = 0
imp_t_g = 0
t_k = 0.8
# training params FIXED
d = 0.1
bs = 32
lr = 1e-6
init_eps = 1
activation_str = "relu"
backprop_method = "ig"
init_attn = 1
h = 256
n_layers = 1
# endregion

# region INTRO
# torch.serialization.add_safe_globals([NoKembModel])

torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset", type=str, nargs="?", default="synthetic", help="Dataset name"
)
parser.add_argument(
    "-temp", action="store_true", help="Enable mode flag", default=False
)
parser.add_argument(
    "-tempphis", action="store_true", help="Enable mode flag", default=False
)
parser.add_argument(
    "-nocheckpoints", action="store_true", help="Enable mode flag", default=False
)
parser.add_argument(
    "-demetra", action="store_true", help="Enable mode flag", default=False
)

args = parser.parse_args()
temp = args.temp
temp_phis = args.tempphis
dataname = args.dataset
demetra = args.demetra
nocheckpoints = args.nocheckpoints

# for absolute reproducibility avoid: AMP, benchmark=True, pin_memory=True, persistent_workers=True

device = get_device()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(SEED)
set_all_possible_seeds(SEED)

print("DEVICE:", device)
# print(f'{torch.cuda.is_available()=}')
print("Assigned GPU(s):", os.environ.get("CUDA_VISIBLE_DEVICES"))
# endregion

# region PATHS
path = "results/ablation_tests/var_concepts/"
if temp:
    path = tempfile.mkdtemp()

results_dir = path + f"{dataname}/"
dataset_info_path = os.path.join(results_dir, "info.txt")
os.makedirs(results_dir, exist_ok=True)

if temp_phis:
    phis_path_og = tempfile.mkdtemp()
else:
    phis_path_og = os.environ["WORK"] + "STELLE/phis/" if demetra else f"phis/"

mpath = (
    os.environ["WORK"] + f"STELLE/" + results_dir
    if demetra
    else results_dir + "checkpoints/"
)
model_path_og = tempfile.mkdtemp() if nocheckpoints else mpath

os.makedirs(model_path_og, exist_ok=True)
# endregion


def get_dataset(dataname):
    if dataname == "synthetic":
        # synthetic dummy dataset for quick testing
        global num_classes
        # time series shaped as (n_samples, n_variables, series_length)
        X_full_train = np.random.randn(n_train, n_vars, series_length).astype(
            np.float32
        )
        X_full_test = np.random.randn(n_test, n_vars, series_length).astype(np.float32)

        # string labels to go through convert_labels_to_numeric
        labels = [f"class_{i}" for i in range(num_classes)]
        y_train = [labels[i % num_classes] for i in range(n_train)]
        y_test = [labels[i % num_classes] for i in range(n_test)]
        pass
    else:
        X_full_train, y_train, metadata = load_classification(
            dataname, split="train", return_metadata=True
        )
        X_full_test, y_test, _ = load_classification(
            dataname, split="test", return_metadata=True
        )
        num_classes = len(metadata["class_values"])

    keep = remove_redundant_variables(X_full_train)
    X_train = X_full_train[:, keep, :]
    X_test = X_full_test[:, keep, :]

    # some datasets dont have numeric labels
    numeric_labels, label_to_number_map = convert_labels_to_numeric(y_train)
    y_train = np.asarray(numeric_labels).astype(np.int64)
    y_test = np.asarray([label_to_number_map[label] for label in y_test]).astype(
        np.int64
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.2, random_state=SEED
    )

    with open(dataset_info_path, "w") as f:
        f.write(f"dataname: {dataname}\n")
        f.write(f"X_train.shape: {X_train.shape}\n")
        f.write(f"X_val.shape: {X_val.shape}\n")
        f.write(f"X_test.shape: {X_test.shape}\n")
        f.write(f"num_classes: {num_classes}\n")
        f.write(f"train_subset: {np.bincount(y_train)}\n")
        f.write(f"val_subset: {np.bincount(y_val)}\n")
        f.write(f"test_subset: {np.bincount(y_test)}\n")

        # also print to console
        print(f"dataname: {dataname}")
        print(f"X_train.shape: {X_train.shape}")
        print(f"X_val.shape: {X_val.shape}")
        print(f"X_test.shape: {X_test.shape}")
        print(f"num_classes: {num_classes}")
        print(f"train_subset: {np.bincount(y_train)}")
        print(f"val_subset: {np.bincount(y_val)}")
        print(f"test_subset: {np.bincount(y_test)}")

    train_subset = TrajectoryDataset(
        trajectories=X_train,
        labels=y_train,
        dataname=dataname,
        label_map=label_to_number_map,
        num_classes=num_classes,
    )

    val_subset = TrajectoryDataset(
        trajectories=X_val,
        labels=y_val,
        dataname=dataname,
        label_map=label_to_number_map,
        num_classes=num_classes,
    )

    test_subset = TrajectoryDataset(
        trajectories=X_test,
        labels=y_test,
        dataname=dataname,
        label_map=label_to_number_map,
        num_classes=num_classes,
    )

    train_subset.normalize()
    val_subset.normalize(train_subset.mean, train_subset.std)
    test_subset.normalize(train_subset.mean, train_subset.std)

    trainloader = DataLoader(
        train_subset,
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        worker_init_fn=SEED,
        generator=g,
    )

    testloader = DataLoader(
        test_subset,
        batch_size=bs * 2,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=SEED,
        generator=g,
    )

    valloader = DataLoader(
        val_subset,
        batch_size=bs * 2,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=SEED,
        generator=g,
    )
    return trainloader, testloader, valloader

def set_kernels_and_concepts(train_subset):
    global nvars_formulae, creation_mode
    nvars = train_subset.num_variables
    points = train_subset.num_time_points
    if nvars < nvars_formulae:
        warnings.warn(
            f"{dataname} has {nvars} variables. Attempting to create formulae with more. Setting =."
        )
        nvars_formulae = nvars
    phis_path = phis_path_og +  f"{creation_mode}/{nvars_formulae}_fvars/{nvars}_nvars/t_{t}/"
    os.makedirs(phis_path, exist_ok=True)
    
    creation_mode = 0 if creation_mode == "all" else 1
    if nvars == 1:
        creation_mode = 0
        warnings.warn(
            f"Dataset {dataname} has 1 variable, thus, creation_mode = 0 or 1 is the same case. Collapsing to 0."
        )
        
    mu = BaseMeasure(device=device)
    sampler = STLFormulaGenerator(  # ha points = 100
        max_variables=nvars_formulae,  # max number of vars in a formula
    )
    stlkernel = StlKernel(  # voglio che abbia points = 100
        mu,
        varn=nvars,
        samples=samples,
        newstl=True,
        vectorize=True,  # points=points,  signals=signals,
        normalize=normalize,
        exp_kernel=exp_kernel,
    )
    kernel = TrajectoryKernel(
        mu,
        varn=nvars,
        points=points,
        samples=samples,
        newstl=True,
        normalize=normalize,
        exp_kernel=exp_kernel,
        exp_rhotau=exp_rhotau,
        normalize_rhotau=normalize_rhotau,
        # signals=signals,
    )
    
    formula_manager = FormulaManager(nvars, sampler, stlkernel, pll, t, nvars_formulae, device=device)
    concepts, rhos1, selfk1, total_time = formula_manager.get_formulae(creation_mode, dim_concepts, phis_path, 'concepts', SEED)
    
    scaledconcepts = train_subset.time_scaling(concepts)
    kernel.phis = scaledconcepts
    kernel.rhos_phi = rhos1
    kernel.selfk_phi = selfk1
    
    gc.collect()
    torch.cuda.empty_cache()
    return kernel

def train_test_model(args):
    
    (kernel, trainloader, valloader, testloader) = args
    
    def validate(loader):
        print('Validating...')
        start_time = time()
        results = model.validate(loader, by_class_stats=False)
        total_time = time() - start_time
        print(
            f"Acc: {results['accuracy']:.1f} Wacc: {results['weighted_acc']:.1f}, took {total_time:.0f}s."
        )
        return results, total_time
    
    def save(items, model_path, checkpath): # (model.cpu(), bestepoch, traintime, accuracy, w_accuracy, modelvaltime)
        if os.path.exists(model_path):
            os.remove(model_path)
        torch.save(items, model_path)
        print(f"Trained model saved to {model_path}")

        # remove now useless checkpoint
        if os.path.exists(checkpath):
            os.remove(checkpath)    
    
    def load_model_if_exists(path):
        return None
    
    modelconfig = ModelConfig(
        kernel = kernel,
        concepts = kernel.phis,
        num_classes=trainloader.dataset.num_classes,
        initial_epsilon=init_eps,
        initial_attention_scale=init_attn,
        dropout=d,
        parallel_workers=pll,
        hidden_size=h,
        num_layers=n_layers
    )
    model = ConceptBasedModel(modelconfig).to(device)  
    
    # create a compact, human-readable unique id (dataset, seed, timestamp, random suffix)
    model_id = f"seed_{SEED}_{lr}_{init_attn}_{h}_{n_layers}_t{t}_{dim_concepts}_{creation_mode}_f{nvars_formulae}"
    # attach to model for later reference and debugging
    print(f"Model ID: {model_id}")
    model_path_ev = model_path_og + f"{model_id}.pt"
    checkpath = model_path_ev[:-3] + "_checkpoint.pt"
    
    modelload = load_model_if_exists(model_path_ev)
    model = modelload if modelload is not None else model
    
    if modelload is None:
        param_groups = [
            # Main parameters
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not n.endswith("_strength")
                ]
            },
            # Regularization controllers (higher LR)
            {
                "params": [
                    p for n, p in model.named_parameters() if n.endswith("_strength")
                ],
                "lr": lr * 10,
            },  # higher than default
        ]
        optimizer = optim.Adam(param_groups, lr=lr)
        print("\nStarting model training...\n")
        
        _, bestepoch, traintime = model.train_model(optimizer, trainloader, 
            valloader=valloader,
            num_epochs=int(epochs),
            verbose=verbose,
            checkpath=checkpath,
            patience=patience,
            check_frequency=cf,
            val_every_n_epochs=val_every_n_epochs,)
        
        print(f"\nBest epoch: {bestepoch}, took {traintime:.0f}s. Testing the model...")
        testresults, testtime = validate(testloader)
        save((model.cpu(), bestepoch, traintime, testresults, testtime), model_path_ev, checkpath)
        model = model.to(device)
        
        results = {
            'best_epoch': bestepoch,
            **testresults,
            'test_time': testtime}
                
    return model, model_path_ev, results

def compute_explanations(args): 
    (model_path_ev, trainloader, testloader, model) = args
    
    explanation_layer = model.output_activation.to(device)
    trajbyclass = trainloader.dataset.split_by_class()
    local_explanations_true_pred = []
    
    for i in ['true', 'pred']:
        start_time = time()
        
        explpath = model_path_ev[:-3] + f"_local_explanations_{i}.pickle"
        
        local_explanations = model.get_explanations(
            x=testloader.dataset.trajectories,
            y_true= testloader.dataset.labels if i=='true' else None,
            trajbyclass=trajbyclass,
            layer=explanation_layer,
            t_k=t_k,
            method=backprop_method,
        )
        for e in local_explanations:
            e.generate_explanation(
                improvement_threshold=imp_t_l, enable_postprocessing=True
            )
        local_explanations_time = time() - start_time

        local_explanations_true_pred.append(local_explanations)
        with open(explpath, "wb") as f:
            pickle.dump((local_explanations, local_explanations_time), f)
        print(f"Saved local explanations ({i}) to {explpath}")

    local_metrics = get_local_metrics(local_explanations_true_pred, testloader)

    print(local_metrics)
    print()
    # global
    globpath = model_path_ev[:-3] + "_global_explanations.pickle"
    start_time = time()

    global_explanations = get_training_explanations(model, trainloader, explanation_layer,
                    backprop_method,
                    imp_t_l,
                    imp_t_g,
                    t_k)
    global_explanations_time = time() - start_time
    with open(globpath, 'wb') as f:
        pickle.dump((global_explanations, global_explanations_time), f)
    print(f"Saved global explanations to {globpath}")
    
    global_metrics = get_global_metrics(global_explanations)
    
    del model
    return local_metrics, global_metrics


def main():
    trainloader, valloader, testloader = get_dataset(dataname)
    
    # from here it depends from concepts details
    kernel = set_kernels_and_concepts(trainloader.dataset)
    
    args = (kernel, trainloader, valloader, testloader)
    
    model, model_path_ev, accuracy_results = train_test_model(args)
    
    args_explanations = (model_path_ev, trainloader, testloader, model)
    
    local_metrics, global_metrics = compute_explanations(args_explanations)
    
    result_raw = {}
    # merge dictionaries; later ones override earlier keys on conflict
    if isinstance(accuracy_results, dict):
        result.update(accuracy_results)
    else:
        result_raw["accuracy_results"] = accuracy_results

    if isinstance(local_metrics, dict):
        result.update(local_metrics)
    else:
        result_raw["local_metrics"] = local_metrics

    if isinstance(global_metrics, dict):
        result.update(global_metrics)
    else:
        result_raw["global_metrics"] = global_metrics

    result = {**result_raw["accuracy_results"],
              **result_raw["local_metrics"],
              **result_raw["global_metrics"]}
    
    print("\n\nOK")

if __name__ == "__main__":
    main()
