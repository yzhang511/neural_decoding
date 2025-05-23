"""Example script for running single-session reduced-rank model with hyperparameter sweep.
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
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
from utils.data_loader_utils import SingleSessionDataModule
from models.decoders import ReducedRankDecoder, MLPDecoder, LSTMDecoder, SeanMLPDecoder
from utils.eval_utils import eval_model
from utils.sweep_utils import tune_decoder
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config

BINSIZE = 0.02
LENGTH = 2.
CLASSIFICATION = ["choice"]
REGRESSION = ["wheel-speed", "whisker-motion-energy", "prior", "finger_vel_dim_0", "finger_vel_dim_1"]

"""
-----------
USER INPUTS
-----------
"""
ap = argparse.ArgumentParser()
ap.add_argument("--base_path", type=str, default="/scratch/bcxj/hlyu/RRR/")
ap.add_argument("--eid", type=str)
ap.add_argument("--target", type=str, default="choice", choices=CLASSIFICATION+REGRESSION)
ap.add_argument("--region", type=str, default="all")
ap.add_argument("--method", type=str, default="linear", choices=["linear", "reduced_rank", "mlp", "lstm"])
ap.add_argument("--n_workers", type=int, default=1)
ap.add_argument("--search", action="store_true")
ap.add_argument("--use_nlb", action="store_true")
ap.add_argument("--bin_size", type=int, default=5)
ap.add_argument("--fold_idx", type=int, default=0)
args = ap.parse_args()

OUTPUT_SIZE_LOOKUP = {
    "choice": 2, 
    "prior": 1, 
    "wheel-speed": int(LENGTH/BINSIZE), 
    "whisker-motion-energy": int(LENGTH/BINSIZE),
    "pupil-diameter": int(LENGTH/BINSIZE),
    "finger_vel_dim_0": int(0.6/(args.bin_size/1000)),
    "finger_vel_dim_1": int(0.6/(args.bin_size/1000)),
}


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
save_path = Path(args.base_path)/config.dirs.output_dir/args.eid/args.target/args.method/args.region
ckpt_path = Path(args.base_path)/config.dirs.checkpoint_dir/args.eid/args.target/args.method/args.region 
os.makedirs(save_path, exist_ok=True)
os.makedirs(ckpt_path, exist_ok=True)

model_class = args.method

print(f"Decode {args.target} from session: {args.eid}")


"""
--------
DECODING
--------
"""
print(f'Launch single-session {model_class} decoder:')

# set up model configs
search_space = config.copy()
search_space["eid"] = args.eid
search_space["target"] = args.target
search_space["region"] = args.region if args.region != "all" else "all"
search_space["model"]["output_size"] = OUTPUT_SIZE_LOOKUP[args.target]
search_space["training"]["device"] = torch.device(
    "cuda" if np.logical_and(torch.cuda.is_available(), config.training.device == "gpu") else "cpu"
)
search_space["data"]["use_nlb"] = True if args.use_nlb else False
search_space["data"]["bin_size"] = args.bin_size
search_space["data"]["fold_idx"] = args.fold_idx

# set up for hyperparameter sweep    
if args.search:

    search_space["optimizer"]["lr"] = tune.loguniform(1e-4, 1e-2)
    search_space["optimizer"]["weight_decay"] = tune.loguniform(1e-3, 1.)

    from itertools import combinations
    def generate_mlp_hyperparams(possible_sizes=[256, 128, 64, 32, 16]):
        hyperparams = []
        for length in range(2, len(possible_sizes)):
            for combo in combinations(possible_sizes, length):
                if all(combo[i] > combo[i+1] for i in range(len(combo)-1)):
                    hyperparams.append(f"({', '.join(map(str, combo))})")
        return hyperparams
    
    if model_class == "reduced_rank":
        search_space["optimizer"]["lr"] = 1e-2
        search_space["optimizer"]["weight_decay"] = 1
        # search_space["reduced_rank"]["temporal_rank"] = tune.grid_search(list(range(2, config.tuner.num_samples)))
        search_space["reduced_rank"]["temporal_rank"] = tune.grid_search(list(range(2,  OUTPUT_SIZE_LOOKUP[args.target]+1)))
        search_space["tuner"]["num_epochs"] = config.training.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
    elif model_class == "lstm":
        search_space["lstm"]["lstm_n_layers"] = tune.randint(1, 3)
        search_space["lstm"]["lstm_hidden_size"] = tune.choice([32, 64, 128, 256, 512])
        search_space["lstm"]["mlp_hidden_size"] = tune.choice(
            generate_mlp_hyperparams(possible_sizes=[256, 128, 64, 32, 16])
        )
        search_space["lstm"]["drop_out"] = tune.uniform(0.1, 0.3)
        search_space["tuner"]["num_epochs"] = config.training.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
    elif model_class == "mlp":
        if not args.use_nlb:
            search_space["mlp"]["mlp_hidden_size"] = tune.choice(
                generate_mlp_hyperparams(possible_sizes=[512, 256, 128, 64, 32, 16])
            )
            search_space["mlp"]["drop_out"] = tune.uniform(0.1, 0.3)
        else:
            search_space["mlp"]["n_layers"] = tune.choice(list(range(1, 11)))
            search_space["mlp"]["hidden_dim"] = tune.choice(list(range(50, 601)))
            search_space["mlp"]["drop_out"] = tune.uniform(0.1, 0.5)
        search_space["tuner"]["num_epochs"] = config.training.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
    else:
        raise NotImplementedError

    def train_func(config):
        dm = SingleSessionDataModule(config)
        dm.update_config()
        dm.setup()

        dm.config["training"]["total_steps"] = dm.config["training"]["num_epochs"] * len(dm.train)

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
        num_samples=config.tuner.num_samples if model_class != "reduced_rank" else 1, 
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
dm = SingleSessionDataModule(best_config)
dm.update_config()
dm.setup()

best_config["training"]["total_steps"] = best_config["training"]["num_epochs"] * len(dm.train)

# init and train model
if model_class == "reduced_rank":
    model = ReducedRankDecoder(best_config)
elif model_class == "lstm":
    model = LSTMDecoder(best_config)
elif model_class == "mlp":
    model = MLPDecoder(best_config) if not args.use_nlb else SeanMLPDecoder(best_config)
elif model_class == "linear":
    from scipy.stats import loguniform
    from sklearn.linear_model import RidgeCV, LogisticRegressionCV, Ridge
    from sklearn.model_selection import GridSearchCV
    if args.target in REGRESSION:
        # model = RidgeCV(
        #     alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]
        # )
        model = GridSearchCV(Ridge(), {"alpha": np.logspace(-4, 4, 9)})
    elif args.target in CLASSIFICATION:
        model = LogisticRegressionCV(
            Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4]
        )
    else:
        raise NotImplementedError
else:
    raise NotImplementedError


if model_class != "linear":
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

    train_dataset, test_dataset = dm.train, dm.test

    if model_class == "reduced_rank":
        MODEL_CLASS = ReducedRankDecoder
    elif model_class == "lstm":
        MODEL_CLASS = LSTMDecoder
    elif model_class == "mlp":
        MODEL_CLASS = MLPDecoder if not args.use_nlb else SeanMLPDecoder
    else:
        raise NotImplementedError

    model = MODEL_CLASS.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        config=best_config
    )

"""
----------
EVALUATION
----------
"""
if model_class != "linear":
    model.eval()
    with torch.no_grad():
        metric, test_pred, test_y, test_prob = eval_model(
            train_dataset, 
            test_dataset, 
            model.cpu(), 
            target=config["model"]["target"], 
            model_class=model_class,
            use_nlb=args.use_nlb,
            bin_size=args.bin_size,
            beh_name=args.target
        )
else:
    dm.setup()
    train_dataset, test_dataset = dm.train, dm.test
    metric, test_pred, test_y, test_prob = eval_model(
        train_dataset, 
        test_dataset, 
        model, 
        target=config["model"]["target"], 
        model_class=model_class,
        use_nlb=args.use_nlb,
        bin_size=args.bin_size,
        beh_name=args.target
    )

print(f"Decoding results for {args.eid}: ", metric)
res_dict = {
    "test_metric": metric, 
    "test_pred": test_pred, 
    "test_y": test_y,
    "test_prob": test_prob,
}
if not args.use_nlb:
    np.save(save_path/f'{args.eid}.npy', res_dict)
else:
    np.save(save_path/f'{args.eid}_binSize{args.bin_size}_fold{args.fold_idx}.npy', res_dict)
