import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

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

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

seed_everything(0)

"""
-----------
USER INPUTS
-----------
"""

ap = argparse.ArgumentParser()

ap.add_argument("--base_dir", type=str)
ap.add_argument("--eid", type=str)
ap.add_argument("--n_imposters", type=int, default=10)
ap.add_argument("--target", type=str)
ap.add_argument("--temporal_rank", type=int, default=2)
ap.add_argument("--learning_rate", type=float, default=0.001)
ap.add_argument("--weight_decay", type=float, default=0.001)
ap.add_argument("--max_epochs", type=int, default=500)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--lstm_hidden_size", type=int, default=32)
ap.add_argument("--lstm_n_layers", type=int, default=3)
ap.add_argument("--mlp_hidden_size", type=tuple_type, default="(128, 64, 32)")
ap.add_argument("--drop_out", type=float, default=0.)
ap.add_argument("--lr_factor", type=float, default=0.1)
ap.add_argument("--lr_patience", type=int, default=5)
ap.add_argument("--device", type=str, default="cpu")
ap.add_argument("--n_workers", type=int, default=os.cpu_count())
ap.add_argument("--tune_max_epochs", type=int, default=100)
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
    'weight_decay': tune.grid_search([1e-1, 1e-3]),
    'learning_rate': tune.grid_search([5e-3, 1e-3]),
    'batch_size': tune.grid_search([8, 16]),
    'eid': args.eid,
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
--------
DECODING
--------
"""

for imposter_id in range(-1, args.n_imposters):

    print(f'Decode imposter {imposter_id} for session {args.eid}:')
    print('----------------------------------------------------')
    
    imposter_config = base_config.copy()

    if imposter_id == -1:
        imposter_config['data_dir'] = data_dir
    else:
        imposter_config['data_dir'] = imposter_dir
        imposter_config['imposter_id'] = imposter_id

    def save_results(model_type, r2, test_pred, test_y):
        res_dict = {'r2': r2, 'pred': test_pred, 'target': test_y}
        save_path = res_dir / args.eid / args.target / model_type 
        os.makedirs(save_path, exist_ok=True)
        np.save(save_path / f'imposter_{imposter_id}.npy', res_dict)
        print(f'{args.target} test R2: ', r2)

    for model_type in ['ridge', 'reduced-rank', 'lstm', 'mlp']:

        print(f'Launch {model_type} decoder:')
        print('----------------------------------------------------')

        if model_type == "ridge":
            dm = SingleSessionDataModule(imposter_config)
            dm.setup()
            alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
            model = GridSearchCV(Ridge(), {"alpha": alphas})
            r2, test_pred, test_y = eval_model(dm.train, dm.test, model, model_type=model_type, plot=False)
            save_results(model_type, r2, test_pred, test_y)
            continue

        def train_func(config):
            dm = SingleSessionDataModule(config)
            dm.setup()
            if model_type == "reduced-rank":
                model = ReducedRankDecoder(dm.config)
            elif model_type == "lstm":
                model = LSTMDecoder(dm.config)
            elif model_type == "mlp":
                model = MLPDecoder(dm.config)
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
            search_space = imposter_config.copy()
            search_space['temporal_rank'] = tune.grid_search([2, 5, 10])
        elif model_type == "lstm":
            search_space = imposter_config.copy()
            search_space['lstm_hidden_size'] = tune.grid_search([32, 64])
            search_space['lstm_n_layers'] = tune.grid_search([1, 3, 5])
            search_space['mlp_hidden_size'] = tune.grid_search([(64, 32), (64,), (32,)])
        elif model_type == "mlp":
            search_space = imposter_config.copy()
            search_space['mlp_hidden_size'] = tune.grid_search([(256, 128, 64), (512, 256, 128, 64)])
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
            max_epochs=best_config['max_epochs'], callbacks=[checkpoint_callback], enable_progress_bar=False
        )
        dm = SingleSessionDataModule(best_config)
        dm.setup()
        
        if model_type == "reduced-rank":
            model = ReducedRankDecoder(best_config)
        elif model_type == "lstm":
            model = LSTMDecoder(best_config)
        elif model_type == "mlp":
            model = MLPDecoder(best_config)
        else:
            raise NotImplementedError
        
        trainer.fit(model, datamodule=dm)
        trainer.test(datamodule=dm, ckpt_path='best')

        r2, test_pred, test_y = eval_model(dm.train, dm.test, model, model_type=model_type, plot=False)
        save_results(model_type, r2, test_pred, test_y)

