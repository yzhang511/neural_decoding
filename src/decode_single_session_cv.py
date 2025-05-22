"""Example script for running single-session reduced-rank model with hyperparameter sweep.
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge, LogisticRegression
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from ray import tune
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from loader.base import SingleSessionDataModule
from models.decoders import ReducedRankDecoder, MLPDecoder, LSTMDecoder
from utils.eval import eval_model
from utils.sweep import tune_decoder
from utils.utils import set_seed
from utils.config import config_from_kwargs, update_config

BINSIZE = 0.01
REGRESSION = ["running_speed", "gaze", "pupil"]
CLASSIFICATION = ["gabors", "static_gratings", "drifting_gratings", "flashes"]
LENGTH_LOOKUP = {
    "flashes": 0.2,
    "gabors": 0.2, 
    "static_gratings": 0.2, 
    "drifting_gratings": 1., 
    "running_speed": 1., 
    "gaze": 1., 
    "pupil": 1.
}
OUTPUT_SIZE_LOOKUP = {
    "flashes": 2,
    "gabors": 3, 
    "static_gratings": 6, 
    "drifting_gratings": 8, 
    "running_speed": int(1/BINSIZE), 
    "gaze": int(1/BINSIZE),
    "pupil": int(1/BINSIZE),
}

"""
-----------
USER INPUTS
-----------
"""
ap = argparse.ArgumentParser()
ap.add_argument("--base_path", type=str, default="./")
ap.add_argument("--session_id", type=str)
ap.add_argument("--target", type=str, default="gabors", choices=REGRESSION+CLASSIFICATION)
ap.add_argument("--region", type=str, default="all")
ap.add_argument("--method", type=str, default="linear", choices=["linear"])
ap.add_argument("--n_workers", type=int, default=1)
args = ap.parse_args()


"""
-------
CONFIGS
-------
"""
kwargs = {"model": "include:src/configs/decoder.yaml"}
config = config_from_kwargs(kwargs)
config = update_config("src/configs/decoder.yaml", config)

if args.target in REGRESSION:
    config = update_config("src/configs/reg_trainer.yaml", config)
elif args.target in CLASSIFICATION:
    config = update_config("src/configs/clf_trainer.yaml", config)
else:
    raise NotImplementedError

set_seed(config.seed)

config["dirs"]["data_dir"] = Path(args.base_path)/config.dirs.data_dir
save_path = Path(args.base_path)/config.dirs.output_dir/args.session_id/args.target/args.method/args.region
ckpt_path = Path(args.base_path)/config.dirs.checkpoint_dir/args.session_id/args.target/args.method/args.region 
os.makedirs(save_path, exist_ok=True)
os.makedirs(ckpt_path, exist_ok=True)

model_class = args.method

print(f"Decode {args.target} from session: {args.session_id}")


"""
--------
DECODING
--------
"""
print(f"Launch single-session {model_class} decoder:")

# set up model configs
search_space = config.copy()
search_space["session_id"] = args.session_id
search_space["target"] = args.target
search_space["length"] = LENGTH_LOOKUP[args.target]
search_space["model"]["output_size"] = OUTPUT_SIZE_LOOKUP[args.target]
search_space["region"] = args.region if args.region != "all" else "all"
search_space["training"]["device"] = torch.device(
    "cuda" if np.logical_and(torch.cuda.is_available(), config.training.device == "gpu") else "cpu"
)
    
best_config = search_space

# set up data loader
dm = SingleSessionDataModule(best_config)
dm.update_config()

# init and train model
from scipy.stats import loguniform
param_dist = loguniform(1e-4, 1e4)
if args.target in REGRESSION:
    model = RandomizedSearchCV(
        Ridge(), 
        param_distributions={"alpha": param_dist},
        n_iter=config.tuner.num_samples, cv=5, random_state=config.seed
    )
elif args.target in CLASSIFICATION:
    model = RandomizedSearchCV(
        LogisticRegression(max_iter=1000), 
        param_distributions={"C": param_dist},
        n_iter=config.tuner.num_samples, cv=5, random_state=config.seed
    )

"""
----------
EVALUATION
----------
"""
dm.setup()
train_dataset, val_dataset, test_dataset = dm.train, dm.val, dm.test

train_x, train_y = [], []
for (x, y, region, eid) in train_dataset:
    train_x.append(x.cpu())
    train_y.append(y.cpu())

val_x, val_y = [], []
for (x, y, region, eid) in val_dataset:
    val_x.append(x.cpu())
    val_y.append(y.cpu())

test_x, test_y = [], []
for (x, y, region, eid) in test_dataset:
    test_x.append(x.cpu())
    test_y.append(y.cpu())

train_x, train_y = torch.stack(train_x), torch.stack(train_y)
test_x, test_y = torch.stack(test_x), np.stack(test_y)
val_x, val_y = torch.stack(val_x), torch.stack(val_y)

all_x = np.concatenate([train_x.numpy(), val_x.numpy(), test_x.numpy()], axis=0)
all_y = np.concatenate([train_y, val_y, test_y], axis=0)

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=config.seed)

target = args.target

all_preds_ordered = np.zeros_like(all_y)
all_probs_ordered = np.zeros((len(all_y), OUTPUT_SIZE_LOOKUP[target])) if target in CLASSIFICATION else None
all_ys_ordered = np.zeros_like(all_y)

# Perform 5-fold CV
for fold, (train_idx, test_idx) in enumerate(kf.split(all_x)):
    
    print(f"Training fold {fold+1}/5")
    
    # Split data for this fold
    train_fold_x = all_x[train_idx]
    train_fold_y = all_y[train_idx]
    test_fold_x = all_x[test_idx]
    test_fold_y = all_y[test_idx]

    # Reshape and fit model
    if target in CLASSIFICATION:
        model.fit(train_fold_x.reshape((train_fold_x.shape[0], -1)), train_fold_y)
        test_fold_prob = model.predict_proba(test_fold_x.reshape((test_fold_x.shape[0], -1)))
        test_fold_pred = test_fold_prob.argmax(1)
        # Store probabilities in original order
        all_probs_ordered[test_idx] = test_fold_prob
    elif target in REGRESSION:
        model.fit(train_fold_x.reshape((train_fold_x.shape[0], -1)), train_fold_y)
        test_fold_pred = model.predict(test_fold_x.reshape((test_fold_x.shape[0], -1)))
    
    # Store predictions and true values in original order
    if OUTPUT_SIZE_LOOKUP[target] == 1:
        all_preds_ordered[test_idx] = test_fold_pred.reshape(-1, 1)
        all_ys_ordered[test_idx] = test_fold_y.reshape(-1, 1)
    else:
        all_preds_ordered[test_idx] = test_fold_pred
        all_ys_ordered[test_idx] = test_fold_y
 
res_dict = {
    "all_pred": all_preds_ordered, 
    "all_y": all_ys_ordered,
    "all_prob": all_probs_ordered,
}
np.save(save_path/f'{args.eid}_cv.npy', res_dict)

print(f"Finished decoding for session {args.session_id}!")
    
