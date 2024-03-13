import os
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
from shared_decoding.utils.ibl_data_loaders import SingleSessionDataModule
from shared_decoding.models.neural_models import ReducedRankDecoder, MLPDecoder, LSTMDecoder, eval_model
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
ap.add_argument("--target", type=str)
ap.add_argument("--n_pc_components", type=int, default=10)
ap.add_argument("--temporal_rank", type=int, default=2)
ap.add_argument("--learning_rate", type=float, default=0.001)
ap.add_argument("--weight_decay", type=float, default=0.001)
ap.add_argument("--max_epochs", type=int, default=500)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--lr_factor", type=float, default=0.1)
ap.add_argument("--lr_patience", type=int, default=5)
ap.add_argument("--device", type=str, default="cpu")
ap.add_argument("--n_workers", type=int, default=4)
ap.add_argument("--tune_max_epochs", type=int, default=50)
ap.add_argument("--tune_n_samples", type=int, default=1)

args = ap.parse_args()

base_dir = Path(args.base_dir)
data_dir = base_dir / 'data'
model_dir = base_dir / 'models'
res_dir = base_dir / 'results'

for path in [data_dir, model_dir, res_dir]:
    os.makedirs(path, exist_ok=True)

DEVICE = torch.device('cuda' if np.logical_and(torch.cuda.is_available(), args.device == 'gpu') else 'cpu')

base_config = {
    'data_dir': data_dir,
    'weight_decay': tune.grid_search([0.5, 0.1, 1e-3]),
    'learning_rate': tune.grid_search([5e-3, 1e-3]),
    'batch_size': 8,
    'imposter_id': None,
    'target': args.target,
    'drop_out': 0.,
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

for comp_idx in range(-1, args.n_pc_components):

    print(f'Decode PC component {comp_idx} for session {args.eid}:')
    print('----------------------------------------------------')

    configs = []
    for eid in eids:
        config = base_config.copy()
        config['eid'] = eid
        configs.append(config)
    
    def save_results(eid, model_type, training_type, r2, test_pred, test_y):
        res_dict = {'r2': r2, 'pred': test_pred, 'target': test_y}
        save_path = res_dir / eid / args.target / training_type + '_' + model_type 
        os.makedirs(save_path, exist_ok=True)
        np.save(save_path / f'comp_{comp_idx}.npy', res_dict)
        print(f'{eid} {args.target} test R2: ', r2)

        
    print(f'Launch {training_type} {model_type} decoder:')
    print('----------------------------------------------------')

    def train_func(configs):
        
        if comp_idx != -1:
            dm = MultiSessionDataModule(eids, configs, comp_idxs=[comp_idx])
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
        search_space = configs.copy()
        for idx in range(len(search_space)):
            search_space[idx]['temporal_rank'] = tune.grid_search([5, 10, 15, 20, 25])
    else:
        raise NotImplementedError

    results = tune_decoder(
        train_func, search_space, use_gpu=False, max_epochs=search_space['tune_max_epochs'], 
        num_samples=search_space['tune_n_samples'], num_workers=search_space['n_workers']
    )
    
    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_config = best_result.config['train_loop_config']

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', mode='min', dirpath=model_dir
    )
    
    trainer = Trainer(
        max_epochs=best_config['max_epochs'], callbacks=[checkpoint_callback], enable_progress_bar=True
    )
    
    if comp_idx != -1:
        dm = MultiSessionDataModule(eids, configs, comp_idxs=[comp_idx])
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
        dm.train, dm.test, model, model_type=model_type, training_type=training_type, plot=False
    )
    for eid_idx, eid in eids:
        save_results(
            eid, model_type, training_type, 
            r2_lst[eid_idx], test_pred_lst[eid_idx], test_y_lst[eid_idx]
        )

  