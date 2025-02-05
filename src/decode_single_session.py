"""Example script for running single-session reduced-rank model with hyperparameter sweep.
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LogisticRegression
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
from loader.base import SingleSessionDataModule
from models.decoders import ReducedRankDecoder, MLPDecoder, LSTMDecoder
from utils.eval import eval_model
from utils.sweep import tune_decoder
from utils.utils import set_seed
from utils.config import config_from_kwargs, update_config

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
    "gabors": 2, 
    "static_gratings": 6, 
    "drifting_gratings": 8, 
    "running_speed": 1, 
    "gaze": 1, 
    "pupil": 1
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
ap.add_argument("--method", type=str, default="linear", choices=["linear", "reduced_rank", "mlp", "lstm"])
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
save_path = Path(args.base_path)/config.dirs.output_dir/args.target/args.method/args.region
ckpt_path = Path(args.base_path)/config.dirs.checkpoint_dir/args.target/args.method/args.region 
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
search_space["output_size"] = OUTPUT_SIZE_LOOKUP[args.target]
search_space["region"] = args.region if args.region != "all" else None
search_space["training"]["device"] = torch.device(
    "cuda" if np.logical_and(torch.cuda.is_available(), config.training.device == "gpu") else "cpu"
)

# set up for hyperparameter sweep
if model_class == "linear":
    dm = SingleSessionDataModule(search_space)
    dm.setup()
    if args.target in REGRESSION:
        model = GridSearchCV(Ridge(), {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]})
    elif args.target in CLASSIFICATION:
        model = GridSearchCV(LogisticRegression(), {"C": [1, 1e1, 1e2, 1e3, 1e4]})
    else:
        raise NotImplementedError
else:
    # search_space["optimizer"]["lr"] = tune.grid_search([1e-2, 1e-3])
    # search_space["optimizer"]["weight_decay"] = tune.grid_search([1, 1e-1, 1e-2, 1e-3])
    search_space["optimizer"]["lr"] = tune.grid_search([1e-3])
    search_space["optimizer"]["weight_decay"] = tune.grid_search([1e-2])
    
    if model_class == "reduced_rank":
        # search_space["reduced_rank"]["temporal_rank"] = tune.grid_search([2, 5, 10, 15])
        search_space["reduced_rank"]["temporal_rank"] = tune.grid_search([2])
        search_space["tuner"]["num_epochs"] = 10 # 500
        search_space["training"]["num_epochs"] = 10 # 800
    elif model_class == "lstm":
        search_space["lstm"]["lstm_hidden_size"] = tune.grid_search([128, 64])
        search_space["lstm"]["lstm_n_layers"] = tune.grid_search([1, 3, 5])
        search_space["lstm"]["drop_out"] = tune.grid_search([0., 0.2, 0.4, 0.6])
        search_space["tuner"]["num_epochs"] = 250
        search_space["training"]["num_epochs"] = 250
    elif model_class == "mlp":
        search_space["mlp"]["drop_out"] = tune.grid_search([0., 0.2, 0.4, 0.6])
        search_space["tuner"]["num_epochs"] = 250
        search_space["training"]["num_epochs"] = 250
    else:
        raise NotImplementedError

    def train_func(config):
        dm = SingleSessionDataModule(config)
        dm.setup()
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
        num_samples=config.tuner.num_samples, 
        num_workers=args.n_workers
    )
    best_result = results.get_best_result(metric=config.tuner.metric, mode=config.tuner.mode)
    best_config = best_result.config["train_loop_config"]

    print("Best model config:")
    print(best_config)
    
    # set up trainer
    checkpoint_callback = ModelCheckpoint(
        monitor=config.training.metric, 
        mode=config.training.mode, 
        dirpath=ckpt_path
    )
    trainer = Trainer(
        max_epochs=config.training.num_epochs, 
        callbacks=[checkpoint_callback], 
        enable_progress_bar=config.training.enable_progress_bar
    )

    # set up data loader
    dm = SingleSessionDataModule(best_config)
    dm.setup()

    # init and train model
    if model_class == "reduced_rank":
        model = ReducedRankDecoder(best_config)
    elif model_class == "lstm":
        model = LSTMDecoder(best_config)
    elif model_class == "mlp":
        model = MLPDecoder(best_config)
    else:
        raise NotImplementedError
    
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path="best")
    metrics = trainer.test(datamodule=dm, ckpt_path="best")[0]
    metric = metrics["test_metric"]


"""
----------
EVALUATION
----------
"""
metric, test_pred, test_y = eval_model(
    dm.train, 
    dm.test, 
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
        
