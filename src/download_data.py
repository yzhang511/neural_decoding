import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from one.api import ONE
from datasets import DatasetDict, DatasetInfo
from utils.ibl_data_utils import (
    prepare_data, 
    select_brain_regions, 
    list_brain_regions, 
    bin_spiking_data,
    bin_behaviors,
    align_data,
)
from utils.dataset_utils import create_dataset, upload_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default=None)
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--n_workers", type=int, default=1)
args = ap.parse_args()

SEED = 42
np.random.seed(SEED)

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', 
    password='international', silent=True, 
    cache_dir = args.base_path
)

params = {
    "interval_len": 2, 
    "binsize": 0.02, 
    "single_region": False,
    "align_time": 'stimOn_times', 
    "time_window": (-.5, 1.5), 
    "fr_thresh": 0.5
}

beh_names = ['choice', 'reward', 'block', 'wheel-speed', 'whisker-motion-energy']

DYNAMIC_VARS = list(filter(lambda x: x not in ["choice", "reward", "block"], beh_names))

if args.eid is not None:
    include_eids = [args.eid]
else:
    with open("../data/repro_ephys_release.txt") as file:
        include_eids = [line.rstrip().replace("'", "") for line in file]

print(f"Preprocess a total of {len(include_eids)} EIDs.")

for eid_idx, eid in enumerate(include_eids):

    print('==========================')
    print(f'Preprocess session {eid}:')

    # Load and preprocess data
    neural_dict, behave_dict, meta_dict, trials_dict, _ = prepare_data(
        one, eid, params, n_workers=args.n_workers
    )
    regions, beryl_reg = list_brain_regions(neural_dict, **params)
   
    region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)

    bin_spikes, clusters_used_in_bins = bin_spiking_data(
        region_cluster_ids, 
        neural_dict, 
        trials_df=trials_dict["trials_df"], 
        n_workers=args.n_workers, 
        **params
    )
    print(f"Binned Spike Data: {bin_spikes.shape}")

    bin_beh, beh_mask = bin_behaviors(
        one, 
        eid, 
        DYNAMIC_VARS, 
        trials_df=trials_dict["trials_df"], 
        allow_nans=True, 
        n_workers=args.n_workers, 
        **params,
    )

    try:
        bin_beh["prior"] = np.load(Path(args.base_path)/"prior_localization"/eid/"priors.npy", allow_pickle=True)
        assert len(bin_beh["prior"]) == len(bin_spikes)
    except:
        bin_beh["prior"] = bin_beh["block"]

    try:
        align_bin_spikes, align_bin_beh, align_bin_lfp, _, bad_trial_idxs = align_data(
            bin_spikes, 
            bin_beh, 
            None, 
            list(bin_beh.keys()), 
            trials_dict["trials_mask"], 
        )
    except ValueError as e:
        print(f"Skip EID {eid} due to error: {e}")
        continue

    if "whisker-motion-energy" not in align_bin_beh:
        logging.info(f"Skip EID {eid} due to missing whisker data.")
        continue

    print("Spike Data Shape: ", align_bin_spikes.shape)

    # Partition dataset (train: 0.7 val: 0.1 test: 0.2)
    num_trials = len(align_bin_spikes)
    trial_idxs = np.random.choice(np.arange(num_trials), num_trials, replace=False)
    train_idxs = trial_idxs[:int(0.7*num_trials)]
    val_idxs = trial_idxs[int(0.7*num_trials):int(0.8*num_trials)]
    test_idxs = trial_idxs[int(0.8*num_trials):]

    train_beh, val_beh, test_beh = {}, {}, {}
    for beh in align_bin_beh.keys():
        train_beh.update({beh: align_bin_beh[beh][train_idxs]})
        val_beh.update({beh: align_bin_beh[beh][val_idxs]})
        test_beh.update({beh: align_bin_beh[beh][test_idxs]})
    
    train_dataset = create_dataset(
        align_bin_spikes[train_idxs], 
        eid, 
        params,
        meta_data=meta_dict,
        binned_behaviors=train_beh, 
        binned_lfp=None if align_bin_lfp is None else align_bin_lfp[train_idxs]
    )
    val_dataset = create_dataset(
        align_bin_spikes[val_idxs], 
        eid, 
        params,
        meta_data=meta_dict,
        binned_behaviors=val_beh, 
        binned_lfp=None if align_bin_lfp is None else align_bin_lfp[val_idxs]
    )
    test_dataset = create_dataset(
        align_bin_spikes[test_idxs], 
        eid, 
        params,
        meta_data=meta_dict,
        binned_behaviors=test_beh, 
        binned_lfp=None if align_bin_lfp is None else align_bin_lfp[test_idxs]
    )

    # Create dataset
    partitioned_dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset}
    )
    print(partitioned_dataset)

    # Cache dataset
    save_path = Path(args.base_path)/"ibl_aligned"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    partitioned_dataset.save_to_disk(f'{save_path}/{eid}')

    np.save(save_path/eid/"train_trial_idxs.npy", train_idxs)
    np.save(save_path/eid/"val_trial_idxs.npy", val_idxs)
    np.save(save_path/eid/"test_trial_idxs.npy", test_idxs)

    print(f'Downloaded session {eid}.')
    print(f'Progress: {eid_idx+1} / {len(include_eids)} sessions downloaded.')
