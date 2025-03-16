"""Example script for running multi-session reduced-rank model with hyperparameter sweep.
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from ray import tune
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from utils.data_loader_utils import MultiSessionDataModule
from models.decoders import MultiSessionReducedRankDecoder
from utils.eval_utils import eval_multi_session_model
from utils.sweep_utils import tune_decoder
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config

BINSIZE = 0.02
LENGTH = 2.
CLASSIFICATION = ["choice"]
REGRESSION = ["wheel-speed", "whisker-motion-energy", "pupil-diameter", "prior"]

OUTPUT_SIZE_LOOKUP = {
    "choice": 2, 
    "prior": 1, 
    "wheel-speed": int(LENGTH/BINSIZE), 
    "whisker-motion-energy": int(LENGTH/BINSIZE),
    "pupil-diameter": int(LENGTH/BINSIZE),
}

"""
-----------
USER INPUTS
-----------
"""
ap = argparse.ArgumentParser()
ap.add_argument("--base_path", type=str, default="./")
ap.add_argument("--repo_path", type=str, default="/burg/stats/users/yz4123/neural_decoding")
ap.add_argument("--target", type=str, default="choice", choices=REGRESSION+CLASSIFICATION)
ap.add_argument("--region", type=str, default="all")
ap.add_argument("--method", type=str, default="reduced_rank", choices=["reduced_rank"])
ap.add_argument("--search", action="store_true")
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
save_path = Path(args.base_path)/config.dirs.output_dir/"multi-session"/args.target/args.method/args.region
ckpt_path = Path(args.base_path)/config.dirs.checkpoint_dir/"multi-session"/args.target/args.method/args.region
os.makedirs(save_path, exist_ok=True)
os.makedirs(ckpt_path, exist_ok=True)

model_class = args.method

"""
---------
LOAD DATA
---------
"""
with open(Path(args.repo_path)/'data/ibl_session_ids.txt', 'r') as f:
    eids = f.read().splitlines()  # removes newlines
    eids = [eid.strip() for eid in eids if eid.strip()]

print('---------------------------------------------')
print(f'Decode {args.target} from {len(eids)} sessions:')
print(eids)


"""
--------
DECODING
--------
"""
print('---------------------------------------------')
print(f'Launch multi-session {model_class} decoder:')

# set up model configs
search_space = config.copy()
search_space["target"] = args.target
search_space["model"]["output_size"] = OUTPUT_SIZE_LOOKUP[args.target]
search_space["region"] = args.region if args.region != "all" else "all"
search_space["training"]["device"] = torch.device(
    "cuda" if np.logical_and(torch.cuda.is_available(), config.training.device == "gpu") else "cpu"
)

# set up for hyperparameter sweep
if args.search:

    search_space["optimizer"]["lr"] = 0.01
    search_space["optimizer"]["weight_decay"] = 1
    
    if model_class == "reduced_rank":
        search_space["reduced_rank"]["temporal_rank"] = tune.grid_search(list(range(2, config.tuner.num_samples)))
        search_space["tuner"]["num_epochs"] = config.tuner.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
    else:
        raise NotImplementedError
        
    def train_func(config):
        configs = []
        for eid in eids:
            _config = config.copy()
            _config["eid"] = eid
            configs.append(_config)
        
        dm = MultiSessionDataModule(eids, configs)
        dm.update_config()
        dm.setup()
        
        base_config = dm.configs[0].copy()
        base_config['n_units'] = [_config['n_units'] for _config in dm.configs]
        base_config['eid_to_indx'] = {e: i for i,e in enumerate(eids)}
        best_config["training"]["total_steps"] = best_config["training"]["num_epochs"] * len(dm.train)

        if model_class == "reduced_rank":
            model = MultiSessionReducedRankDecoder(base_config)
        else:
            raise NotImplementedError

        trainer = Trainer(
            max_epochs=config["tuner"]["num_epochs"],
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=config["tuner"]["enable_progress_bar"],
            check_val_every_n_epoch=1,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)

    # hyperparameter sweep
    results = tune_decoder(
        train_func, 
        search_space, 
        save_dir=ckpt_path,
        use_gpu=config.tuner.use_gpu, 
        max_epochs=config.tuner.num_epochs, 
        num_samples=1, 
        num_workers=args.n_workers,
        metric=config.tuner.metric,
        mode=config.tuner.mode,
    )
    best_result = results.get_best_result(metric=config.tuner.metric, mode=config.tuner.mode)
    best_config = best_result.config["train_loop_config"]

    print("Best model config:")
    print(best_config)

    torch.save(best_config, ckpt_path / "best_config.pth")


if not args.search:
    best_config = search_space
else:
    best_config = torch.load(ckpt_path / "best_config.pth")

# set up data loader
configs = []
for eid in eids:
    config = best_config.copy()
    config["eid"] = eid
    configs.append(config)

dm = MultiSessionDataModule(eids, configs)
dm.update_config()
dm.setup()

best_config = dm.configs[0].copy()
best_config["n_units"] = [_config["n_units"] for _config in dm.configs]
best_config["eid_to_indx"] = {e: i for i, e in enumerate(eids)}

num_train = 0
for _train_loader in dm.train:
    num_train += len(_train_loader) * best_config["training"]["batch_size"]
best_config["training"]["total_steps"] = best_config["training"]["num_epochs"] * num_train

# init and train model
if model_class == "reduced_rank":
    model = MultiSessionReducedRankDecoder(best_config)
else:
    raise NotImplementedError

model.to(best_config["training"]["device"])

# set up trainer
checkpoint_callback = ModelCheckpoint(
    monitor=best_config["training"]["metric"], 
    mode=best_config["training"]["mode"], 
    dirpath=ckpt_path
)
trainer = Trainer(
    max_epochs=best_config["training"]["num_epochs"], 
    callbacks=[checkpoint_callback], 
    enable_progress_bar=best_config["training"]["enable_progress_bar"],
    check_val_every_n_epoch=1, 
    devices=1, # Use only one GPU
    strategy="auto",  
)

trainer.fit(model, datamodule=dm)

train_dataset, test_dataset = dm.train, dm.test

model = MultiSessionReducedRankDecoder.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    config=best_config
)

"""
----------
EVALUATION
----------
"""
model.eval()
with torch.no_grad():
    metric_lst, test_pred_lst, test_y_lst, test_prob_lst = eval_multi_session_model(
        train_dataset, 
        test_dataset, 
        model.cpu(),
        target=best_config["model"]["target"], 
        configs=configs,
    )

print("Decoding results for each session:")

for eid_idx, eid in enumerate(eids):
    print(f"{eid}: {metric_lst[eid_idx]}")
    res_dict = {
        "test_metric": metric_lst[eid_idx], 
        "test_pred": test_pred_lst[eid_idx], 
        "test_y": test_y_lst[eid_idx],
        "test_prob": test_prob_lst[eid_idx],
    }
    np.save(save_path/f"{eid}.npy", res_dict)
        