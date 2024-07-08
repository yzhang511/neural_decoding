import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer

from shared_decoding.utils.ibl_data_utils import seed_everything
from shared_decoding.utils.ibl_data_loaders import SingleSessionDataModule
from shared_decoding.models.neural_models import ReducedRankDecoder, eval_model

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
ap.add_argument("--target", type=str)
ap.add_argument("--temporal_rank", type=tuple_type, default="(2, 5, 10)")
ap.add_argument("--learning_rate", type=float, default=0.01)
ap.add_argument("--weight_decay", type=float, default=0.1)
ap.add_argument("--max_epochs", type=int, default=500)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--lr_factor", type=float, default=0.1)
ap.add_argument("--lr_patience", type=int, default=5)
ap.add_argument("--device", type=str, default="cpu")
ap.add_argument("--n_workers", type=int, default=os.cpu_count())

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
    'weight_decay': 1e-1,
    'learning_rate': args.learning_rate,
    'batch_size': 8,
    'eid': args.eid,
    'imposter_id': None,
    'target': args.target,
    'lr_factor': 0.1,
    'lr_patience': 5,
    'device': DEVICE,
    'n_workers': args.n_workers,
    'max_epochs': args.max_epochs,
}

"""
--------
DECODING
--------
"""

for rank in args.temporal_rank:

    print(f'Decode {args.target} in session {args.eid} with rank {rank}:')
    print('----------------------------------------------------')

    base_config['temporal_rank'] = rank

    def save_results(rank, r2, test_pred, test_y):
        res_dict = {'r2': r2, 'pred': test_pred, 'target': test_y}
        save_path = res_dir / "sensitivity_analysis" / args.eid / args.target  
        os.makedirs(save_path, exist_ok=True)
        np.save(save_path / f'rank_{rank}.npy', res_dict)
        print(f'{args.target} test R2: ', r2)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', mode='min', dirpath=model_dir
    )
    
    trainer = Trainer(
        max_epochs=base_config['max_epochs'], callbacks=[checkpoint_callback], enable_progress_bar=False
    )
    dm = SingleSessionDataModule(base_config)
    dm.setup()

    model = ReducedRankDecoder(base_config)

    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path='best')

    r2, test_pred, test_y = eval_model(dm.train, dm.test, model, model_type="reduced-rank", plot=False)
    save_results(rank, r2, test_pred, test_y)


