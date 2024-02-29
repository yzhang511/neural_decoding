import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from one.api import ONE
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
from shared_decoding.utils.ibl_data_utils import (
    prepare_data, 
    select_brain_regions, list_brain_regions, 
    bin_spiking_data, bin_behaviors,
    save_data, save_imposter_sessions
)

"""
-----------
USER INPUTS
-----------
"""

ap = argparse.ArgumentParser()

ap.add_argument("--base_dir", type=str)
ap.add_argument("--align_time", type=str, default="firstMovement_times")
ap.add_argument("--trial_start", type=float, default=-0.2)
ap.add_argument("--trial_end", type=float, default=0.8)
ap.add_argument("--binsize", type=float, default=0.02)
ap.add_argument("--single_region", action='store_false', default=False)
ap.add_argument("--eid_idx", type=int, default=1)

args = ap.parse_args()

base_dir = Path(args.base_dir)
data_dir = base_dir/'data'

params = {
    'align_time': args.align_time,
    'time_window': (args.trial_start, args.trial_end),
    'binsize': args.binsize,
    'single_region': args.single_region, # use all available regions
    'interval_len': args.trial_end - args.trial_start,
}

"""
----------
CACHE DATA
-----------
"""

one = ONE(
    base_url='https://openalyx.internationalbrainlab.org', password='international', silent=True
)
ba = AllenAtlas()

freeze_file = '../data/2023_12_bwm_release.csv'
bwm_df = pd.read_csv(freeze_file, index_col=0)

eid = bwm_df.eid[args.eid_idx]

neural_dict, behave_dict, meta_data, trials_data = prepare_data(one, eid, bwm_df, params)

regions, beryl_reg = list_brain_regions(neural_dict, **params)
region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)

binned_spikes, clusters_used_in_bins = bin_spiking_data(
    region_cluster_ids, neural_dict, trials_df=trials_data['trials_df'], **params
)

binned_behaviors, behavior_masks = bin_behaviors(
    one, eid, trials_df=trials_data['trials_df'], allow_nans=True, **params
)

save_data(eid, binned_spikes, binned_behaviors, save_path=data_dir)

