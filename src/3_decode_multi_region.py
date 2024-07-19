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
from utils.data_loaders import MultiRegionDataModule
from models.decoders import MultiRegionReducedRankDecoder
from utils.eval import eval_multi_region_model
from utils.hyperparam_tuning import tune_decoder
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config

ap = argparse.ArgumentParser()
ap.add_argument(
    "--target", type=str, default="choice", 
    choices=["choice", "wheel-speed", "whisker-motion-energy", "pupil-diameter"]
)
ap.add_argument("--method", type=str, default="reduced_rank", choices=["reduced_rank"])
ap.add_argument("--n_workers", type=int, default=1)
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
args = ap.parse_args()

"""
-------
CONFIGS
-------
"""

kwargs = {"model": "include:src/configs/decoder.yaml"}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/decoder.yaml", config)

if args.target in ["wheel-speed", "whisker-motion-energy", "pupil-diameter"]:
    config = update_config("src/configs/reg_trainer.yaml", config)
elif args.target in ['choice']:
    config = update_config("src/configs/clf_trainer.yaml", config)
else:
    raise NotImplementedError

if config.wandb.use:
    import wandb
    wandb.login()
    wandb.init(
        config=config,
        name="train_{}".format(args.method)
    )
set_seed(config.seed)

config["dirs"]["data_dir"] = Path(args.base_path)/config.dirs.data_dir
save_path = Path(args.base_path)/config.dirs.output_dir/args.target/('multi-region-'+args.method) 
ckpt_path = Path(args.base_path)/config.dirs.checkpoint_dir/args.target/('multi-region-'+args.method) 
os.makedirs(save_path, exist_ok=True)
os.makedirs(ckpt_path, exist_ok=True)

"""
---------
LOAD DATA
---------
"""

eids = [
    fname for fname in os.listdir(config.dirs.data_dir) if fname != "downloads"
]

print(eids)

"""
--------
DECODING
--------
"""

model_class = args.method

print('----------------------------------------------------')
print(f'Decode {args.target} from {len(eids)} sessions:')
print(f'Launch multi-region {model_class} decoder:')

search_space = config.copy()
search_space['target'] = args.target
search_space['region'] = 'all' 
search_space['query_region'] = ['CA1', 'PO', 'LP']
search_space['training']['device'] = torch.device(
    'cuda' if np.logical_and(torch.cuda.is_available(), config.training.device == 'gpu') else 'cpu'
)

search_space['optimizer']['lr'] = 1e-2 #tune.grid_search([1e-2, 1e-3])
search_space['optimizer']['weight_decay'] = 1e-1 #tune.grid_search([0, 1e-1, 1e-2, 1e-3, 1e-4])

if model_class == "reduced_rank":
    search_space['temporal_rank'] = 3 #tune.grid_search([2, 5, 10, 15, 20])
    search_space['global_basis_rank'] = 5 #tune.grid_search([5, 10])
    search_space['tuner']['num_epochs'] = 10 #500
    search_space['training']['num_epochs'] = 100 #800
else:
    raise NotImplementedError

best_config = search_space.copy()

print("Best config:")
print(best_config)

# Model training 

checkpoint_callback = ModelCheckpoint(
    monitor=config.training.metric, mode=config.training.mode, dirpath=ckpt_path
)

trainer = Trainer(
    max_epochs=config.training.num_epochs, 
    callbacks=[checkpoint_callback], 
    enable_progress_bar=config.training.enable_progress_bar
)

configs = []
for eid in eids:
    config = best_config.copy()
    config['eid'] = eid
    configs.append(config)
    
dm = MultiRegionDataModule(eids, configs)
dm.list_regions()
regions_dict = dm.regions_dict

configs = []
for eid in eids:
    for region in search_space['query_region']:
        if region in regions_dict[eid]:
            config = best_config.copy()
            config['eid'] = eid
            config['region'] = region
            configs.append(config)

dm = MultiRegionDataModule(eids, configs)
dm.setup()

best_config = dm.configs[0].copy()
best_config['n_units'] = []
best_config['eid_region_to_indx'] = {}
for eid in eids:
    best_config['eid_region_to_indx'][eid] = {}

for idx, _config in enumerate(dm.configs):
    best_config['n_units'].append(_config['n_units'])
    best_config['eid_region_to_indx'][_config['eid']][_config['region']] = idx
    
best_config['n_regions'] = len(search_space['query_region'])
best_config['region_to_indx'] = {r: i for i,r in enumerate(search_space['query_region'])}

print(best_config['eid_region_to_indx'])
    
if model_class == "reduced_rank":
    model = MultiRegionReducedRankDecoder(best_config)
else:
    raise NotImplementedError

trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm, ckpt_path='best')


metric_dict, test_pred_dict, test_y_dict = eval_multi_region_model(
    dm.train, dm.test, model, target=best_config['model']['target'], 
    all_regions=search_space['query_region'], configs=dm.configs,
)

print(metric_dict)
    
if config["wandb"]["use"]:
    wandb.log(
        {"eids": eids, "test_metric": metric_dict}
    )
    wandb.finish()
else:
    for region in metric_dict.keys():
        print(region)
        for eid in metric_dict[region].keys():
            res_dict = {
                'test_metric': metric_dict[region][eid], 
                'test_pred': test_pred_dict[region][eid], 
                'test_y': test_y_dict[region][eid]
            }
            print(f'{eid}: {metric_dict[region][eid]}')
            os.makedirs(save_path/region, exist_ok=True)
            np.save(save_path/region/f'{eid}.npy', res_dict)
        