import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from one.api import ONE
from datasets import DatasetDict
from utils.ibl_data_utils import (
    prepare_data, 
    select_brain_regions, 
    list_brain_regions, 
    bin_spiking_data,
    bin_behaviors,
    align_spike_behavior
)
from utils.dataset_utils import create_dataset, upload_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default=None)
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--fold_idx", type=int, default=5)
ap.add_argument("--n_workers", type=int, default=1)
args = ap.parse_args()

SEED = 42
np.random.seed(SEED)
assert args.fold_idx > 0, "Fold idx must be from 1 to 5."

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', 
    silent=True,
    cache_dir = args.base_path
)
# Trial setup
params = {
    'interval_len': 2, 'binsize': 0.02, 'single_region': False,
    'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)
}

beh_names = ['choice', 'reward', 'stimside', 'wheel-speed', 'whisker-motion-energy']

if args.eid is not None:
    include_eids = [args.eid]
else:
    with open("../data/repro_ephys_release.txt") as file:
        include_eids = [line.rstrip().replace("'", "") for line in file]

print(f"Preprocess a total of {len(include_eids)} EIDs.")

num_neurons = []
for eid_idx, eid in enumerate(include_eids):

    print('==========================')
    print(f'Preprocess session {eid}:')

    # Load and preprocess data
    neural_dict, behave_dict, meta_data, trials_data = prepare_data(one, eid, params, n_workers=args.n_workers)
    regions, beryl_reg = list_brain_regions(neural_dict, **params)
   
    region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)
    binned_spikes, clusters_used_in_bins = bin_spiking_data(
        region_cluster_ids, neural_dict, trials_df=trials_data['trials_df'], n_workers=args.n_workers, **params
    )
    try:
        binned_behaviors, behavior_masks = bin_behaviors(
            one, eid, beh_names[3:], trials_df=trials_data['trials_df'], 
            allow_nans=True, n_workers=args.n_workers, **params
        ) 
        # Ensure neural and behavior data match for each trial
        aligned_binned_spikes, aligned_binned_behaviors = align_spike_behavior(
            binned_spikes, binned_behaviors, beh_names, trials_data['trials_mask']
        )
    except ValueError as e:
        print(e)
        continue

    print("spike data shape: ", aligned_binned_spikes.shape)
    num_neurons.append(aligned_binned_spikes.shape[-1])

    # Partition dataset (train: 0.7 val: 0.1 test: 0.2)
    max_num_trials = len(aligned_binned_spikes)
    trial_idxs = np.random.choice(np.arange(max_num_trials), max_num_trials, replace=False)
    
    num_folds = 5
    fold_size = len(trial_idxs) // num_folds
    folds = [trial_idxs[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]
     
    for fold in range(num_folds):

        if (fold+1) != args.fold_idx:
            continue

        test_idxs = folds[fold]

        train_val_idxs = np.concatenate([folds[i] for i in range(num_folds) if i != fold])

        num_train = int(0.875 * len(train_val_idxs))  # 70% of the total data
        train_idxs = train_val_idxs[:num_train]
        val_idxs = train_val_idxs[num_train:]

        print(f"Fold {fold + 1}:")

    if len(train_idxs) == 0 or len(val_idxs) == 0 or len(test_idxs) == 0:
        print(f"Skip {eid} due to empty set.")
        continue

    is_cls_balance = True
    train_beh, val_beh, test_beh = {}, {}, {}
    for beh in aligned_binned_behaviors.keys():
        train_beh.update({beh: aligned_binned_behaviors[beh][train_idxs]})
        val_beh.update({beh: aligned_binned_behaviors[beh][val_idxs]})
        test_beh.update({beh: aligned_binned_behaviors[beh][test_idxs]})

        if beh in ["choice", "stimside", "reward"]:
            if any(len(np.unique(beh_data)) < 2 for beh_data in \
                    [train_beh[beh], val_beh[beh], test_beh[beh]]):
                is_cls_balance = False
                break
    
    if not is_cls_balance:
        print(f"Skip {eid} due to imbalanced cls distribution.")
        continue
    
    train_dataset = create_dataset(
        aligned_binned_spikes[train_idxs], eid, params, 
        binned_behaviors=train_beh, meta_data=meta_data
    )
    val_dataset = create_dataset(
        aligned_binned_spikes[val_idxs], eid, params, 
        binned_behaviors=val_beh, meta_data=meta_data
    )
    test_dataset = create_dataset(
        aligned_binned_spikes[test_idxs], eid, params, 
        binned_behaviors=test_beh, meta_data=meta_data
    )

    # Create dataset
    partitioned_dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset}
    )
    print(partitioned_dataset)

    # Cache dataset
    save_path = Path(args.base_path)/'ibl_aligned'/f'fold_{args.fold_idx}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    partitioned_dataset.save_to_disk(f'{save_path}/{eid}')

    print(f'Downloaded session {eid}.')
    print(f'Progress: {eid_idx+1} / {len(include_eids)} sessions downloaded.')

