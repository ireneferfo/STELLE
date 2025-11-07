import torch
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
import json
import tempfile
import warnings
from datetime import datetime
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split

# ensure workspace root is on sys.path so the local `STELLE` package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from STELLE.data.dataset_utils import (
    remove_redundant_variables,
    convert_labels_to_numeric,
)
from STELLE.utils import flatten_dict, get_device, set_all_possible_seeds
from STELLE.data.data_generation import generate_synthetic_trajectories
from STELLE.data.base_dataset import TrajectoryDataset
from STELLE.kernels.base_measure import BaseMeasure
from STELLE.kernels.stl_kernel import StlKernel
from STELLE.kernels.trajectory_kernel import TrajectoryKernel
from STELLE.formula_generation.stl_generator import STLFormulaGenerator
from STELLE.formula_generation.formula_manager import FormulaManager
from STELLE.model.model import ConceptBasedModel, ModelConfig
from STELLE.explanations.global_explanation import get_training_explanations
from STELLE.explanations.explanation_metrics import (
    get_local_metrics,
    get_global_metrics,
)


# region FIXED PARAMETERS

# for synthetic data generation
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
epochs = 30
cf = 50  # check frequency
patience = 5
val_every_n_epochs = 1
verbose = 1
logging = False
normalize = False
exp_kernel = False
normalize_rhotau = True
exp_rhotau = True
# concepts FIXED
t = 1  # 0.99
n_vars_formulae = 1
creation_mode = "one"
min_total = 100  # ignore the minimum
imp_t_l = 0
imp_t_g = 0
t_k = 0.8
# training params FIXED
d = 0.1
bs = 32
lr = 1e-4
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
path = "ablation_tests/results/var_dimconcepts/"
if temp:
    path = tempfile.mkdtemp()

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if dataname == 'synthetic': 
    dataname += f'_{n_train}_{n_test}_v{n_vars}_{series_length}_c{num_classes}'

results_dir = path + f"{dataname}/{run_id}/"

dataset_info_path = os.path.join(results_dir, "info.txt")

if temp_phis:
    phis_path_og = tempfile.mkdtemp()
else:
    phis_path_og = os.environ["WORK"] + "STELLE/phis/" if demetra else "phis/"

mpath = (
    os.environ["WORK"] + "STELLE/" + path + f"{dataname}/checkpoints/"
    if demetra
    else path + f"{dataname}/checkpoints/"
)
model_path_og = tempfile.mkdtemp() if nocheckpoints else mpath

os.makedirs(model_path_og, exist_ok=True)
# endregion


def save_run_settings():
    constants = {
        'run_id': run_id,
        'seed': SEED,
        "t": t,
        "nvars_formulae": n_vars_formulae,
        "creation_vars": creation_mode,
        'min_total': min_total,
        "bs": bs,
        "lr": lr,
        "init_attn": init_attn,
        "init_eps": init_eps,
        "d": d,
        "n_layers": n_layers,
        "imp_t_l": imp_t_l,
        "imp_t_g": imp_t_g,
        "t_k": t_k,
        "pll": pll,
        "newstl": "True",
        "normalize": normalize,
        "exponentiate": exp_kernel,
        "normalize_rhotau": normalize_rhotau,
        "exp_rhotau": exp_rhotau,
        "backprop_method": backprop_method,
    }
    
    info_file_path = results_dir + 'run_info.txt'
    
    with open(info_file_path, "w") as file:
        file.write(json.dumps(constants))


def get_dataset(dataname):
    if "synthetic" in dataname:
        global num_classes
        X_full_train, y_train = generate_synthetic_trajectories(n_train, n_vars, series_length, num_classes, SEED)
        X_full_test, y_test = generate_synthetic_trajectories(n_test, n_vars, series_length, num_classes, SEED+1)
        
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
    del X_full_train, X_full_test
    
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
        # print(f"dataname: {dataname}")
        # print(f"X_train.shape: {X_train.shape}")
        # print(f"X_val.shape: {X_val.shape}")
        # print(f"X_test.shape: {X_test.shape}")
        # print(f"num_classes: {num_classes}")
        # print(f"train_subset: {np.bincount(y_train)}")
        # print(f"val_subset: {np.bincount(y_val)}")
        # print(f"test_subset: {np.bincount(y_test)}")

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


def set_kernels_and_concepts(train_subset, dim_concepts):
    global n_vars_formulae, creation_mode
    n_vars = train_subset.num_variables
    points = train_subset.num_time_points
    if n_vars < n_vars_formulae:
        warnings.warn(
            f"{dataname} has {n_vars} variables. Attempting to create formulae with more. Setting =."
        )
        n_vars_formulae = n_vars
    phis_path = (
        phis_path_og + f"{creation_mode}/{n_vars_formulae}_fvars/{n_vars}_n_vars/t_{t}/"
    )
    os.makedirs(phis_path, exist_ok=True)

    creation_mode = 0 if creation_mode == "all" else 1
    if n_vars == 1:
        creation_mode = 0
        warnings.warn(
            f"Dataset {dataname} has 1 variable, thus, creation_mode = 0 or 1 is the same case. Collapsing to 0."
        )

    mu = BaseMeasure(device=device)
    sampler = STLFormulaGenerator(  # ha points = 100
        max_variables=n_vars_formulae,  # max number of vars in a formula
    )
    stlkernel = StlKernel(  # voglio che abbia points = 100
        mu,
        varn=n_vars,
        samples=samples,
        newstl=True,
        vectorize=True,  # points=points,  signals=signals,
        normalize=normalize,
        exp_kernel=exp_kernel,
    )
    kernel = TrajectoryKernel(
        mu,
        varn=n_vars,
        points=points,
        samples=samples,
        newstl=True,
        normalize=normalize,
        exp_kernel=exp_kernel,
        exp_rhotau=exp_rhotau,
        normalize_rhotau=normalize_rhotau,
        # signals=signals,
    )

    formula_manager = FormulaManager(
        n_vars, sampler, stlkernel, pll, t, n_vars_formulae, device=device
    )
    concepts, rhos1, selfk1, total_time = formula_manager.get_formulae(
        creation_mode, dim_concepts, phis_path, "concepts", SEED
    )

    scaledconcepts = train_subset.time_scaling(concepts)
    kernel.phis = scaledconcepts
    kernel.rhos_phi = rhos1
    kernel.selfk_phi = selfk1

    gc.collect()
    torch.cuda.empty_cache()
    return kernel, total_time


def train_test_model(args):
    # return None, None, {}
    (kernel, trainloader, valloader, testloader, dim_concepts) = args

    def validate(loader):
        start_time = time()
        results = model.validate(loader, by_class_stats=False)
        total_time = time() - start_time
        print(
            f"Acc: {results['accuracy']:.1f} Wacc: {results['weighted_acc']:.1f}, took {total_time:.0f}s."
        )
        return results, total_time

    def save(
        items, model_path, checkpath
    ):  # (model.cpu(), bestepoch, traintime, accuracy, w_accuracy, modelvaltime)
        if os.path.exists(model_path):
            os.remove(model_path)
        torch.save(items, model_path)
        print(f"Trained model saved to {model_path}")

        # remove now useless checkpoint
        if os.path.exists(checkpath):
            os.remove(checkpath)

    def load_model_if_exists(path):
        if os.path.exists(path):
            try:
                out = torch.load(path, weights_only=False)
                print(f"\nLoading existing model from {path}.")
                model, bestepoch, traintime,  testresults, testtime = out
                model = model.to(device)
                model.device = device
                return model, bestepoch, traintime, testresults, testtime
            except Exception as e:
                print(f"Failed to load existing model ({e}), will train a new one.")
                return None
        else:
            return None


    modelconfig = ModelConfig(
        kernel=kernel,
        concepts=kernel.phis,
        num_classes=trainloader.dataset.num_classes,
        initial_epsilon=init_eps,
        initial_attention_scale=init_attn,
        dropout=d,
        parallel_workers=pll,
        hidden_size=h,
        num_layers=n_layers,
    )
    model = ConceptBasedModel(modelconfig).to(device)

    # create a compact, human-readable unique id (dataset, seed, timestamp, random suffix)
    model_id = f"seed_{SEED}_{lr}_{init_attn}_{h}_{n_layers}_t{t}_{dim_concepts}_{creation_mode}_f{n_vars_formulae}"
    # attach to model for later reference and debugging
    print(f"Model ID: {model_id}")
    model_path_ev = model_path_og + f"{model_id}.pt"
    checkpath = model_path_ev[:-3] + "_checkpoint.pt"

    modelload = load_model_if_exists(model_path_ev)
    model = modelload if modelload is not None else model

    if modelload is not None:
        model, bestepoch, traintime, testresults, testtime= modelload
    else:
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
        testresults, testtime = validate(testloader)
        save(
            (model.cpu(), bestepoch, traintime, testresults, testtime),
            model_path_ev,
            checkpath,
        )
        model = model.to(device)

    results = {"best_epoch": bestepoch, **testresults, "train_time": traintime, "test_time": testtime}

    return model, model_path_ev, results


def compute_explanations(args):
    return {}, {}
    (model_path_ev, trainloader, testloader, model) = args

    explanation_layer = model.output_activation.to(device)
    trajbyclass = trainloader.dataset.split_by_class()
    local_explanations_true_pred = []

    for i in ["true", "pred"]:
        explpath = model_path_ev[:-3] + f"_local_explanations_{i}.pickle"
        compute = True
        if os.path.exists(explpath):
            try:
                with open(explpath, "rb") as f:
                    local_explanations, local_explanations_time = pickle.read(f)
                compute = False
            except Exception as e:
                print(f"Failed to load existing local explanations - {i} ({e}).")
        
        if compute:
            start_time = time()

            local_explanations = model.get_explanations(
                x=testloader.dataset.trajectories,
                y_true=testloader.dataset.labels if i == "true" else None,
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
    compute = True
    if os.path.exists(globpath):
        try:
            with open(globpath, "rb") as f:
                global_explanations, global_explanations_time = pickle.read(f)
            compute = False
        except Exception as e:
            print(f"Failed to load existing global explanations ({e}).")
    
    if compute:
        start_time = time()

        global_explanations = get_training_explanations(
            model, trainloader, explanation_layer, backprop_method, imp_t_l, imp_t_g, t_k
        )
        global_explanations_time = time() - start_time
        with open(globpath, "wb") as f:
            pickle.dump((global_explanations, global_explanations_time), f)
        print(f"Saved global explanations to {globpath}")

    global_metrics = get_global_metrics(global_explanations)

    del model
    return local_metrics, global_metrics


def main():
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_path_og, exist_ok=True)

    trainloader, valloader, testloader = get_dataset(dataname)
    
    print(f"Run ID: {run_id}\n")
    save_run_settings()

    results = []
    
    # from here it depends from concepts details
    for dim_concepts in [10]: # , 1000, 2000, 5000]:
        kernel, concepts_time = set_kernels_and_concepts(trainloader.dataset, dim_concepts)

        args = (kernel, trainloader, valloader, testloader, dim_concepts)

        model, model_path_ev, accuracy_results = train_test_model(args)

        args_explanations = (model_path_ev, trainloader, testloader, model)

        local_metrics, global_metrics = compute_explanations(args_explanations)
        
        result_raw = {}
        # merge dictionaries; later ones override earlier keys on conflict
        if isinstance(accuracy_results, dict):
            result_raw.update(accuracy_results)
        else:
            result_raw["accuracy_results"] = accuracy_results
        
        if isinstance(local_metrics, dict):
            result_raw.update(local_metrics)
        else:
            result_raw["local_metrics"] = local_metrics

        if isinstance(global_metrics, dict):
            result_raw.update(global_metrics)
        else:
            result_raw["global_metrics"] = global_metrics
        
        result = {
            'dim_concepts': dim_concepts,
            'concepts_time': round(concepts_time,3),
            **result_raw
        }
        
        result = flatten_dict(result)
        keys = result.keys()
        results.append(result)
        
        with open(os.path.join(results_dir, "results.csv"), 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)



if __name__ == "__main__":
    main()
