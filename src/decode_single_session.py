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
ap.add_argument("--session_id", type=str)
ap.add_argument("--target", type=str, default="gabors", choices=REGRESSION+CLASSIFICATION)
ap.add_argument("--region", type=str, default="all")
ap.add_argument("--search", action="store_true")
ap.add_argument("--method", type=str, default="linear", choices=["linear", "reduced_rank", "mlp", "lstm"])
ap.add_argument("--n_workers", type=int, default=1)
ap.add_argument("--eval", action="store_true")
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
    
# set up for hyperparameter sweep
if args.search:

    search_space["optimizer"]["lr"] = tune.loguniform(1e-3, 1e-2)
    search_space["optimizer"]["weight_decay"] = tune.loguniform(0.01, 1.)

    from itertools import combinations
    def generate_mlp_hyperparams(possible_sizes=[256, 128, 64, 32, 16]):
        hyperparams = []
        for length in range(2, len(possible_sizes)):
            for combo in combinations(possible_sizes, length):
                if all(combo[i] > combo[i+1] for i in range(len(combo)-1)):
                    hyperparams.append(f"({', '.join(map(str, combo))})")
        return hyperparams
    
    if model_class == "reduced_rank":
        num_timesteps = int(search_space["length"]/BINSIZE)
        search_space["optimizer"]["lr"] = 0.001 if args.target in CLASSIFICATION else 0.01
        search_space["optimizer"]["weight_decay"] = 1
        search_space["reduced_rank"]["temporal_rank"] = tune.grid_search(list(range(2, 30)))
        search_space["tuner"]["num_epochs"] = config.tuner.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
    elif model_class == "lstm":
        search_space["lstm"]["lstm_n_layers"] = tune.randint(1, 3)
        search_space["lstm"]["lstm_hidden_size"] = tune.choice([32, 64, 128, 256, 512])
        search_space["lstm"]["mlp_hidden_size"] = tune.choice(
            generate_mlp_hyperparams(possible_sizes=[256, 128, 64, 32, 16])
        )
        search_space["lstm"]["drop_out"] = tune.uniform(0.1, 0.3)
        search_space["tuner"]["num_epochs"] = config.tuner.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
    elif model_class == "mlp":
        search_space["mlp"]["mlp_hidden_size"] = tune.choice(
            generate_mlp_hyperparams(possible_sizes=[512, 256, 128, 64, 32, 16])
        )
        search_space["mlp"]["drop_out"] = tune.uniform(0.1, 0.3)
        search_space["tuner"]["num_epochs"] = config.tuner.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
    else:
        raise NotImplementedError

    def train_func(config):
        dm = SingleSessionDataModule(config)
        dm.update_config()

        if model_class == "reduced_rank":
            model = ReducedRankDecoder(dm.config)
        elif model_class == "lstm":
            model = LSTMDecoder(dm.config)
        elif model_class == "mlp":
            model = MLPDecoder(dm.config)
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
        num_samples=1 if model_class == "reduced_rank" else config.tuner.num_samples, 
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
dm = SingleSessionDataModule(best_config)
dm.update_config()

if args.eval:
    best_config = torch.load(ckpt_path / "best_config.pth")

# init and train model
if model_class == "reduced_rank":
    model = ReducedRankDecoder(best_config)
elif model_class == "lstm":
    model = LSTMDecoder(best_config)
elif model_class == "mlp":
    model = MLPDecoder(best_config)
elif model_class == "linear":
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
    else:
        raise NotImplementedError
else:
    raise NotImplementedError


if model_class != "linear":
    if not args.eval:
        model.to(best_config["training"]["device"])

        # set up trainer
        checkpoint_callback = ModelCheckpoint(
            monitor=config.training.metric, 
            mode=config.training.mode, 
            dirpath=ckpt_path
        )
        trainer = Trainer(
            max_epochs=config.training.num_epochs, 
            callbacks=[checkpoint_callback], 
            enable_progress_bar=config.training.enable_progress_bar,
            check_val_every_n_epoch=1 if model_class == "reduced_rank" else 10, # Otherwise too slow
            devices=1, # Use only one GPU
            strategy="auto",  
        )

        trainer.fit(model, datamodule=dm)
        
    else:
        ckpt_file = [f for f in os.listdir(ckpt_path) if f.endswith(".ckpt")][0]
        model.load_state_dict(torch.load(ckpt_path/ckpt_file))
        model.to(best_config["training"]["device"])

"""
----------
EVALUATION
----------
"""

if model_class != "linear":
    train_dataset, test_dataset = dm.train, dm.test
    model.eval()
    with torch.no_grad():
        metric, test_pred, test_y = eval_model(
            train_dataset, 
            test_dataset, 
            model, 
            target=config["model"]["target"], 
            model_class=model_class
        )
else:
    dm.setup()
    train_dataset, test_dataset = dm.train, dm.test
    metric, test_pred, test_y = eval_model(
        train_dataset, 
        test_dataset, 
        model, 
        target=config["model"]["target"], 
        model_class=model_class
    )

print(f"Decoding results for {args.session_id}: ", metric)
res_dict = {
    "test_metric": metric, 
    "test_pred": test_pred, 
    "test_y": test_y,
}
np.save(save_path/f"{args.session_id}.npy", res_dict)
    
