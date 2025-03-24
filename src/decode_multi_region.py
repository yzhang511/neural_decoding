"""Example script for running multi-region reduced-rank model w/o hyperparameter sweep.
"""
import os
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
from utils.sweep_utils import tune_decoder
from utils.data_loader_utils import MultiRegionDataModule
from models.decoders import MultiRegionReducedRankDecoder
from utils.eval_utils import eval_multi_region_model
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config

BINSIZE = 0.02
LENGTH = 2.
CLASSIFICATION = ["choice"]
REGRESSION = ["wheel-speed", "whisker-motion-energy", "prior"]

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
ap.add_argument("--query_region", nargs="+", default=["PO", "LP", "DG", "CA1", "VISa"])
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
save_path = Path(args.base_path)/config.dirs.output_dir/"multi-region"/args.target/args.method
ckpt_path = Path(args.base_path)/config.dirs.checkpoint_dir/"multi-region"/args.target/args.method
os.makedirs(save_path, exist_ok=True)
os.makedirs(ckpt_path, exist_ok=True)

model_class = args.method
query_region = args.query_region


"""
---------
LOAD DATA
---------
"""
# with open(Path(args.repo_path)/'data/region_session_ids.txt', 'r') as f:
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
print(f'Launch multi-region {model_class} decoder:')

# set up model configs
search_space = config.copy()
search_space['target'] = args.target
search_space["model"]["output_size"] = OUTPUT_SIZE_LOOKUP[args.target]
search_space['region'] = 'all' 
search_space['query_region'] = query_region
search_space['training']['device'] = torch.device(
    'cuda' if np.logical_and(torch.cuda.is_available(), config.training.device=='gpu') else 'cpu'
)
search_space['training']['num_epochs'] = 2000
search_space['tuner']['num_epochs'] = 2000
search_space['tuner']['num_samples'] = 30

search_space["optimizer"]["lr"] = 1e-3
search_space["optimizer"]["weight_decay"] = 1

# set up for hyperparameter sweep
if args.search:
    
    if model_class == "reduced_rank":
        search_space["reduced_rank"]["temporal_rank"] = 2
        search_space["reduced_rank"]["global_basis_rank"] = tune.grid_search(list(range(2, config.tuner.num_samples)))
        search_space["tuner"]["num_epochs"] = config.tuner.num_epochs
        search_space["training"]["num_epochs"] = config.training.num_epochs
    else:
        raise NotImplementedError
        
    def train_func(config):

        configs = []
        for eid in eids:
            _config = config.copy()
            _config['eid'] = eid
            configs.append(_config)
        
        dm = MultiRegionDataModule(eids, configs)
        dm.list_regions()  # check all available regions
        regions_dict = dm.regions_dict

        configs = []
        for eid in eids:
            for region in query_region:
                # only load data from sessions containing this region
                if region in regions_dict[eid]:
                    _config = config.copy()
                    _config['eid'] = eid
                    _config['region'] = region
                    configs.append(_config)

        dm = MultiRegionDataModule(eids, configs)
        dm.update_config()
        dm.setup()

        base_config = dm.configs[0].copy()
        base_config['n_units'], base_config['eid_region_to_indx'] = [], {}
        for eid in eids:
            base_config['eid_region_to_indx'][eid] = {}
        # build a dict indexing each session-region combination
        for idx, _config in enumerate(dm.configs):
            base_config['n_units'].append(_config['n_units'])
            base_config['eid_region_to_indx'][_config['eid']][_config['region']] = idx
        # build a dict indexing each brain region
        base_config['region_to_indx'] = {r: i for i,r in enumerate(query_region)}
        base_config['n_regions'] = len(query_region)

        num_train = 0
        for _train_loader in dm.train:
            num_train += len(_train_loader) * base_config["training"]["batch_size"]
        base_config["training"]["total_steps"] = base_config["training"]["num_epochs"] * num_train

        if model_class == "reduced_rank":
            model = MultiRegionReducedRankDecoder(base_config)
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
    devices=1, # use only one GPU
    strategy="auto", 
    ############################## 
    gradient_clip_val=1.0, 
    gradient_clip_algorithm="norm" 
    ############################## 
)

# set up data loader
configs = []
for eid in eids:
    config = best_config.copy()
    config['eid'] = eid
    configs.append(config)

dm = MultiRegionDataModule(eids, configs)
dm.list_regions()  # check all available regions
regions_dict = dm.regions_dict

configs = []
for eid in eids:
    for region in query_region:
        # only load data from sessions containing this region
        if region in regions_dict[eid]:
            config = best_config.copy()
            config['eid'] = eid
            config['region'] = region
            configs.append(config)

dm = MultiRegionDataModule(eids, configs)
dm.update_config()
dm.setup()

np.save(ckpt_path / 'configs.npy', dm.configs)

# init and train model
base_config = dm.configs[0].copy()
base_config['n_units'], base_config['eid_region_to_indx'] = [], {}
for eid in eids:
    base_config['eid_region_to_indx'][eid] = {}
# build a dict indexing each session-region combination
for idx, _config in enumerate(dm.configs):
    base_config['n_units'].append(_config['n_units'])
    base_config['eid_region_to_indx'][_config['eid']][_config['region']] = idx
# build a dict indexing each brain region
base_config['region_to_indx'] = {r: i for i,r in enumerate(query_region)}
base_config['n_regions'] = len(query_region)

num_train = 0
for _train_loader in dm.train:
    num_train += len(_train_loader) * base_config["training"]["batch_size"]
base_config["training"]["total_steps"] = base_config["training"]["num_epochs"] * num_train

print("Index for region and session:")
print(base_config['eid_region_to_indx'])

print("Index for region:")
print(base_config['region_to_indx'])

print("N units:")
print(base_config['n_units'])
 
if model_class == "reduced_rank":
    model = MultiRegionReducedRankDecoder(base_config)
else:
    raise NotImplementedError

model.to(base_config["training"]["device"])

trainer.fit(model, datamodule=dm)

model = MultiRegionReducedRankDecoder.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    config=base_config
)

train_dataset, test_dataset = dm.train, dm.test

"""
----------
EVALUATION
----------
"""
model.eval()
with torch.no_grad():
    metric_dict, test_pred_dict, test_y_dict, test_prob_dict = eval_multi_region_model(
        train_dataset, 
        test_dataset, 
        model.cpu(),
        target=base_config['model']['target'], 
        all_regions=query_region, 
        configs=dm.configs,
    )
    
for region in metric_dict.keys():
    print(region)
    for eid in metric_dict[region].keys():
        print(f"{eid}: {metric_dict[region][eid]}")
        res_dict = {
            'test_metric': metric_dict[region][eid], 
            'test_pred': test_pred_dict[region][eid], 
            'test_y': test_y_dict[region][eid],
            'test_prob': test_prob_dict[region][eid],
        }
        os.makedirs(save_path/region, exist_ok=True)
        np.save(save_path/region/f'{eid}.npy', res_dict)
        
