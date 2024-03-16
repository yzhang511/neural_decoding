import os
os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer

from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from shared_decoding.utils.ibl_data_utils import seed_everything
from shared_decoding.utils.ibl_data_loaders import MultiSessionDataModule
from shared_decoding.models.neural_models import MultiSessionReducedRankDecoder, eval_multi_session_model
from shared_decoding.utils.hyperparam_tuning import tune_decoder

from ray import tune

seed_everything(0)

"""
-----------
USER INPUTS
-----------
"""

ap = argparse.ArgumentParser()

ap.add_argument("--base_dir", type=str)
ap.add_argument("--n_imposters", type=int, default=10)
ap.add_argument("--target", type=str)
ap.add_argument("--smooth_behavior", action='store_false', default=True)
ap.add_argument("--temporal_rank", type=int, default=2)
ap.add_argument("--learning_rate", type=float, default=0.001)
ap.add_argument("--weight_decay", type=float, default=0.001)
ap.add_argument("--max_epochs", type=int, default=500)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--lr_factor", type=float, default=0.1)
ap.add_argument("--lr_patience", type=int, default=5)
ap.add_argument("--device", type=str, default="cpu")
ap.add_argument("--n_workers", type=int, default=4)
ap.add_argument("--tune_max_epochs", type=int, default=35)
ap.add_argument("--tune_n_samples", type=int, default=1)

args = ap.parse_args()

base_dir = Path(args.base_dir)
data_dir = base_dir / 'data'
imposter_dir = base_dir/'imposter'
model_dir = base_dir / 'models'
res_dir = base_dir / 'results'

for path in [data_dir, imposter_dir, model_dir, res_dir]:
    os.makedirs(path, exist_ok=True)

DEVICE = torch.device('cuda' if np.logical_and(torch.cuda.is_available(), args.device == 'gpu') else 'cpu')

base_config = {
    'data_dir': data_dir,
    'weight_decay': tune.grid_search([0.1, 1e-3]),
    'learning_rate': 5e-3,
    'batch_size': 8,
    'imposter_id': None,
    'target': args.target,
    'lr_factor': 0.1,
    'lr_patience': 5,
    'device': DEVICE,
    'n_workers': args.n_workers,
    'max_epochs': args.max_epochs,
    'tune_max_epochs': args.tune_max_epochs,
    'tune_n_samples': args.tune_n_samples
}

"""
---------
LOAD DATA
---------
"""

eids = [fname.split('.')[0] for fname in os.listdir(data_dir) if fname.endswith('npz')]

"""
--------
DECODING
--------
"""

training_type = 'multi-sess'
model_type = 'reduced-rank'

for imposter_id in range(-1, args.n_imposters):

    print(f'Decode imposter {imposter_id} for {len(eids)} sessions:')
    print('----------------------------------------------------')
    
    imposter_config = base_config.copy()

    if imposter_id == -1:
        imposter_config['data_dir'] = data_dir
    else:
        imposter_config['data_dir'] = imposter_dir
        imposter_config['imposter_id'] = imposter_id

    configs = []
    for eid in eids:
        config = imposter_config.copy()
        config['eid'] = eid
        configs.append(config)

    def save_results(eid, model_type, training_type, r2, test_pred, test_y):
        res_dict = {'r2': r2, 'pred': test_pred, 'target': test_y}
        save_path = res_dir / eid / args.target / (training_type + '-' + model_type) 
        os.makedirs(save_path, exist_ok=True)
        np.save(save_path / f'imposter_{imposter_id}.npy', res_dict)
        print(f'{eid} {args.target} test R2: ', r2)

    print(f'Launch {training_type} {model_type} decoder:')
    print('----------------------------------------------------')

    def train_func(base_config):
        
        configs = []
        for eid in eids:
            config = base_config.copy()
            config['eid'] = eid
            configs.append(config)
            
        if args.smooth_behavior:
            dm = MultiSessionDataModule(eids, configs, comp_idxs=[0,1])
        else:
            dm = MultiSessionDataModule(eids, configs)
        dm.setup()
        base_config = dm.configs[0].copy()
        base_config['n_units'] = [config['n_units'] for config in dm.configs]
            
        if model_type == "reduced-rank":
            model = MultiSessionReducedRankDecoder(base_config)
        else:
            raise NotImplementedError
    
        trainer = Trainer(
            max_epochs=config['max_epochs'],
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)

    if model_type == "reduced-rank":
        search_space = base_config.copy()
        search_space['temporal_rank'] = tune.grid_search([5, 15, 25])
    else:
        raise NotImplementedError

    results = tune_decoder(
        train_func, search_space, use_gpu=False, max_epochs=base_config['tune_max_epochs'], 
        num_samples=base_config['tune_n_samples'], num_workers=base_config['n_workers']
    )
    
    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_config = best_result.config['train_loop_config']

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', mode='min', dirpath=model_dir
    )
    
    trainer = Trainer(
        max_epochs=best_config['max_epochs'], callbacks=[checkpoint_callback], enable_progress_bar=True
    )

    configs = []
    for eid in eids:
        config = best_config.copy()
        config['eid'] = eid
        configs.append(config)
    
    if args.smooth_behavior:
        dm = MultiSessionDataModule(eids, configs, comp_idxs=[0,1])
    else:
        dm = MultiSessionDataModule(eids, configs)
    dm.setup()
    best_config = dm.configs[0].copy()
    best_config['n_units'] = [config['n_units'] for config in dm.configs]
    
    if model_type == "reduced-rank":
        model = MultiSessionReducedRankDecoder(best_config)
    else:
        raise NotImplementedError
    
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path='best')

    r2_lst, test_pred_lst, test_y_lst = eval_multi_session_model(
        dm.train, dm.test, model, plot=False
    )
    for eid_idx, eid in enumerate(eids):
        save_results(
            eid, model_type, training_type, 
            r2_lst[eid_idx], test_pred_lst[eid_idx], test_y_lst[eid_idx]
        )
