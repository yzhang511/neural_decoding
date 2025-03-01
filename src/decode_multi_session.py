"""Example script for running multi-session reduced-rank model with hyperparameter sweep.
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
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
from loader.base import MultiSessionDataModule
from models.decoders import MultiSessionReducedRankDecoder
from utils.eval import eval_multi_session_model
from utils.sweep import tune_decoder
from utils.utils import set_seed
from utils.config import config_from_kwargs, update_config

BINSIZE = 0.01
REGRESSION = ["running_speed", "gaze", "pupil"]
CLASSIFICATION = ["gabors", "static_gratings", "drifting_gratings"]
LENGTH_LOOKUP = {
    "gabors": 0.2, 
    "static_gratings": 0.2, 
    "drifting_gratings": 1., 
    "running_speed": 1., 
    "gaze": 1., 
    "pupil": 1.
}
OUTPUT_SIZE_LOOKUP = {
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
ap.add_argument("--target", type=str, default="gabors", choices=REGRESSION+CLASSIFICATION)
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
eids = [fname.replace(".pkl", "") for fname in os.listdir(config.dirs.data_dir) if fname.endswith(".pkl")]

# Filter out sessions with missing train / val / test data
invalid_eids = [
    fname.replace(".pkl", "") for fname in os.listdir(config.dirs.data_dir) 
    if fname.endswith(".pkl") and not all(
        os.path.isdir(
            os.path.join(Path(args.base_path)/"datasets/cached", fname.replace(".pkl", ""), args.target, args.region, split)
        )
        for split in ["train", "val", "test"]
    )
]

if invalid_eids:
    print(f"Found {len(invalid_eids)} sessions with missing train or val or test data:")
    for invalid_eid in invalid_eids:
        print(f"- {invalid_eid}")

# Filter out invalid sessions from eids
eids = [eid for eid in eids if eid not in invalid_eids]

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
search_space["length"] = LENGTH_LOOKUP[args.target]
search_space["model"]["output_size"] = OUTPUT_SIZE_LOOKUP[args.target]
search_space["region"] = args.region if args.region != "all" else "all"
search_space["training"]["device"] = torch.device(
    "cuda" if np.logical_and(torch.cuda.is_available(), config.training.device == "gpu") else "cpu"
)

# set up for hyperparameter sweep
if args.search:

    search_space["optimizer"]["lr"] = 0.001 if args.target in CLASSIFICATION else 0.01
    search_space["optimizer"]["weight_decay"] = 1
    
    if model_class == "reduced_rank":
        search_space["reduced_rank"]["temporal_rank"] = tune.grid_search(list(range(2, 30)))
        search_space["tuner"]["num_epochs"] = config.tuner.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
    else:
        raise NotImplementedError
        
    def train_func(config):
        configs = []
        for eid in eids:
            _config = config.copy()
            _config["session_id"] = eid
            configs.append(_config)
        
        dm = MultiSessionDataModule(eids, configs)
        dm.update_config()
        dm.setup()
        
        base_config = dm.configs[0].copy()
        base_config['num_units'] = [_config['num_units'] for _config in dm.configs]
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

# set up data loader
configs = []
for eid in eids:
    config = best_config.copy()
    config["session_id"] = eid
    configs.append(config)

dm = MultiSessionDataModule(eids, configs)
dm.update_config()
dm.setup()

best_config = dm.configs[0].copy()
best_config["num_units"] = [_config["num_units"] for _config in dm.configs]
best_config["eid_to_indx"] = {e: i for i, e in enumerate(eids)}
best_config["training"]["total_steps"] = best_config["training"]["num_epochs"] * len(dm.train)

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


"""
----------
EVALUATION
----------
"""
model.eval()
with torch.no_grad():
    metric_lst, test_pred_lst, test_y_lst = eval_multi_session_model(
        train_dataset, 
        test_dataset, 
        model, 
        target=best_config["model"]["target"], 
    )

print("Decoding results for each session:")

for eid_idx, eid in enumerate(eids):
    print(f"{eid}: {metric_lst[eid_idx]}")
    res_dict = {
        "test_metric": metric_lst[eid_idx], 
        "test_pred": test_pred_lst[eid_idx], 
        "test_y": test_y_lst[eid_idx]
    }
    np.save(save_path/f"{eid}.npy", res_dict)
        