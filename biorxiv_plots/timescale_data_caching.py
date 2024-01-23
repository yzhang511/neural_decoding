import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np

import torch

from one.api import ONE
from iblatlas.atlas import AllenAtlas

from side_info_decoding.utils import (
    set_seed, 
    load_data_from_pids, 
    sliding_window_over_trials
)

if __name__ == "__main__":
    
    # -- args
    ap = argparse.ArgumentParser()
    g = ap.add_argument_group("Data Input/Output")
    g.add_argument("--atlas_level", type=int, default=7)
    g.add_argument("--roi_idx", type=int)
    g.add_argument("--out_path", type=str)
    g.add_argument("--n_sess", default=20, type=int)
    args = ap.parse_args()
    
    seed = 666
    set_seed(seed)
    
    ba = AllenAtlas()
    regions = np.unique(ba.regions.acronym[ba.regions.level == args.atlas_level])
    
    print("=================")
    print("Downloading data for regions: ", regions)
    
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org", mode='remote'
    )
    roi = regions[args.roi_idx]
    
    print("=================")
    print(f"Downloading data in region {roi} ..")
    
    pids = one.search_insertions(atlas_acronym=[roi], query_type='remote')
    pids = list(pids)[:args.n_sess]
    
    # load choice
    neural_dict, choice_dict = load_data_from_pids(
        pids,
        brain_region=roi.lower(),
        behavior="choice",
        data_type="all_ks",
        n_t_bins = 40,
    )
    available_pids = list(neural_dict.keys())
    
    # load contrast
    _, contrast_dict = load_data_from_pids(
        pids,
        brain_region=roi.lower(),
        behavior="contrast",
        data_type="good_ks",
        n_t_bins = 40,
    )

    print("=================")
    print(f"Downloaded {len(available_pids)} PIDs in region {roi} ..")
    
    for _, pid in enumerate(available_pids):
        xs, ys = neural_dict[pid], choice_dict[pid]
        n_trials, n_units, n_t_bins = xs.shape
        if n_units < 5:
            continue
        xs = sliding_window_over_trials(xs, half_window_size=0).squeeze()
        ys = sliding_window_over_trials(ys, half_window_size=0).squeeze()
        xs, ys = torch.tensor(xs), torch.tensor(ys)
        
        contrast_dict[pid] = np.nan_to_num(contrast_dict[pid], 0)
        contrast_dict[pid].T[0] *= -1
        contrast = contrast_dict[pid].sum(1)
        
        contrast_mask_dict = {}
        for lvl in np.unique(np.abs(contrast)):
            contrast_mask_dict.update(
                {lvl: np.argwhere(contrast == lvl).flatten()}
            )
            
        path = args.out_path/roi
        if not os.path.exists(path):
            os.makedirs(path)
            
        data_dict = {}
        data_dict.update({'contrast': contrast})
        data_dict.update({'contrast_mask': contrast_mask_dict})
        data_dict.update({'meta':
            {"n_trials": n_trials, "n_units": n_units, "n_t_bins": n_t_bins}
        })
        xs_per_lvl, ys_per_lvl = {}, {}
        xs_per_lvl.update({"all": xs})
        ys_per_lvl.update({"all": ys})
        for lvl in np.unique(np.abs(contrast)):
            try:
                xs_per_lvl.update({lvl: xs[contrast_mask_dict[lvl]]})
                ys_per_lvl.update({lvl: ys[contrast_mask_dict[lvl]]})
            except:
                continue
        data_dict.update({'neural_contrast': xs_per_lvl})
        data_dict.update({'choice_contrast': ys_per_lvl})
        np.save(path/f"pid_{pid}.npy", data_dict)
        
    print("=================")
    print(f"Successfully cached all data!")
    
    