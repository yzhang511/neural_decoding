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
    align_spike_behavior,
    create_intervals, 
)
from utils.dataset_utils import create_dataset, upload_dataset

from reproducible_ephys_functions import filter_recordings
from fig_PCA.fig_PCA_load_data import load_dataframe

params = {
    "interval_len": 2, 
    "binsize": 0.02, 
    "single_region": False,
    "align_time": "stimOn_times", 
    "time_window": (-.5, 1.5)
}

beh_names = ["wheel-speed", "whisker-motion-energy", "pupil-diameter"]

def main(one, base_path, fold_idx, include_eids, n_workers):

    assert fold_idx > 0, "Fold idx must be from 1 to 5."

    print(f"Preprocess a total of {len(include_eids)} EIDs.")

    for eid_idx, eid in enumerate(include_eids):

        print(f"Process session {eid}...")

        neural_dict, _, meta_data, _ = prepare_data(one, eid, None, params, n_workers=n_workers)
        regions, beryl_reg = list_brain_regions(neural_dict, **params)
        region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)

        intervals = create_intervals(
            start_time=0, 
            end_time=neural_dict["spike_times"].max(), 
            interval_len=params["interval_len"]
        )

        binned_spikes, clusters_used_in_bins = bin_spiking_data(
            region_cluster_ids, 
            neural_dict, 
            intervals=intervals, 
            n_workers=n_workers, 
            **params
        )

        try:
            binned_behaviors, behavior_masks = bin_behaviors(
                one, eid, beh_names, intervals=intervals, allow_nans=True, n_workers=n_workers, **params
            )
            aligned_binned_spikes, aligned_binned_behaviors = align_spike_behavior(
                binned_spikes, binned_behaviors, beh_names
            )
        except ValueError as e:
            print(e)
            continue

        max_num_trials = len(aligned_binned_spikes)
        trial_idxs = np.random.choice(np.arange(max_num_trials), max_num_trials, replace=False)
        train_idxs = trial_idxs[:int(0.7 * max_num_trials)]
        val_idxs = trial_idxs[int(0.7 * max_num_trials):int(0.8 * max_num_trials)]
        test_idxs = trial_idxs[int(0.8 * max_num_trials):]

        train_beh, val_beh, test_beh = {}, {}, {}
        for beh in aligned_binned_behaviors.keys():
            train_beh[beh] = aligned_binned_behaviors[beh][train_idxs]
            val_beh[beh] = aligned_binned_behaviors[beh][val_idxs]
            test_beh[beh] = aligned_binned_behaviors[beh][test_idxs]

        train_dataset = create_dataset(
            aligned_binned_spikes[train_idxs], None, eid, params, 
            binned_behaviors=train_beh, meta_data=meta_data
        )
        val_dataset = create_dataset(
            aligned_binned_spikes[val_idxs], None, eid, params, 
            binned_behaviors=val_beh, meta_data=meta_data
        )
        test_dataset = create_dataset(
            aligned_binned_spikes[test_idxs], None, eid, params, 
            binned_behaviors=test_beh, meta_data=meta_data
        )

        partitioned_dataset = DatasetDict({
            "train": train_dataset, "val": val_dataset, "test": test_dataset
        })
        print(partitioned_dataset)

        save_path = Path(base_path)/"cached_ibl_data"
        save_path.mkdir(parents=True, exist_ok=True)
        partitioned_dataset.save_to_disk(f"{save_path}/{eid}")

        print(f"Cached session {eid}.")
        print(f"Progress: {eid_idx + 1} / {len(include_eids)} sessions downloaded.")


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_path", type=str, default="/scratch/bcxj/hlyu/RRR/")
    ap.add_argument("--eid", type=str)
    ap.add_argument("--fold_idx", type=int, default=5)
    ap.add_argument("--n_workers", type=int, default=1)
    args = ap.parse_args()

    SEED = 42
    np.random.seed(SEED)

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org", 
        password="international", 
        silent=True,
        cache_dir=args.base_path
    )

    if args.eid is not None:
        include_eids = [args.eid]
    else:
        concat_df = load_dataframe()
        concat_df = filter_recordings(concat_df, min_regions=0)
        include_eids = np.unique(concat_df.eid)

    main(
        one, args.base_path, args.fold_idx, include_eids,args.n_workers
    )

