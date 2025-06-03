"""Load data, processes it, save it."""

import argparse
import logging
import os
import pickle
from typing import Dict

import random
import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
from tqdm import tqdm
from utils.utils import set_seed

logging.basicConfig(level=logging.INFO)


def extract_spikes(session, session_id):

    units = session.get_units()
    spiketimes_dict = session.spike_times
    chan_ids = session.get_units().peak_channel_id.values

    chan_index = session.get_channels().index.values
    chan_region = session.get_channels().structure_acronym.values
    region_dict = dict(zip(chan_index, chan_region))

    region_index = [region_dict[chan_id] for chan_id in chan_ids]
    print(set(region_index))

    spikes = []
    unit_index = []
    unit_meta = []
    for i, unit_id in enumerate(spiketimes_dict.keys()):
        metadata = units.loc[unit_id]
        unit_name = f"{session_id}/{unit_id}"

        spiketimes = spiketimes_dict[unit_id]
        spikes.append(spiketimes)
        unit_index.append([i] * len(spiketimes))

        unit_meta.append(
            {
                "count": len(spiketimes),
                "id": unit_name,
                "unit_number": i,
            }
        )

    spikes = np.concatenate(spikes)
    unit_index = np.concatenate(unit_index)

    # convert unit metadata to a Data object
    unit_meta_df = pd.DataFrame(unit_meta)
    units = {
        "unit_index": unit_index, 
        "unit_meta_df": unit_meta_df,
        "region_index": region_index,
    }

    return spikes, units


def extract_rewards(session):

    stimulus_presentations = session.stimulus_presentations
    stim_blocks = stimulus_presentations.stimulus_block.values
    stim_block_len = [sum(stim_blocks == stim) for stim in set(stim_blocks)]
    chosen_stim_block = np.unique(stim_blocks)[np.argmax(stim_block_len)]
    print(f"Chosen stimulus block {chosen_stim_block} with length {np.max(stim_block_len)}")

    mask = stimulus_presentations.stimulus_block == chosen_stim_block
    stim_block_start = stimulus_presentations.start_time[mask]
    stim_block_end = stimulus_presentations.end_time[mask]
    print(f"Chosen stimulus block start from {stim_block_start.min()} to {stim_block_end.max()}")

    licks = session.licks
    licks_timestamps = licks.timestamps.values.astype(np.float32)

    print(licks_timestamps)
    min_duration = np.min(np.diff(licks_timestamps))
    max_duration = np.max(np.diff(licks_timestamps))
    print(f"Min & Max lick duration: {min_duration:.3f} & {max_duration:.3f} seconds")

    start_time = np.floor(licks_timestamps.min())  
    end_time = np.ceil(licks_timestamps.max())    

    def count_consecutive_runs(arr):
        """Returns total number of runs of 1s and 0s"""
        return np.sum(arr[1:] != arr[:-1])

    # Try multiple bin sizes and pick one with fewest transitions (most consecutive 1s and 0s)
    bin_sizes = np.linspace(min_duration, 2.0, 50)  # try from min gap up to 2 second
    best_score = float('inf')
    best_bin_size = None
    best_binary = None

    for bs in bin_sizes:
        bins = np.arange(start_time, end_time + bs, bs)
        idx = np.digitize(licks_timestamps, bins)
        binary = np.zeros(len(bins))
        binary[np.unique(idx)] = 1
        score = count_consecutive_runs(binary)
        
        if score < best_score:
            best_bins = bins
            best_score = score
            best_bin_size = bs
            best_binary = binary
    print("Best bin size for max consecutive 1s and 0s:", best_bin_size)

    orientations = np.array(best_binary, dtype=np.int32)
    start_times = np.array(bins[:-1], dtype=np.float32)
    end_times = np.array(bins[1:], dtype=np.float32)
    output_timestamps = (
        start_times + (end_times - start_times) / 2
    )
    print(output_timestamps)
    print(orientations)

    return {
        "start": start_times,
        "end": end_times,
        "orientation": orientations,  # (N,)
        "timestamps": output_timestamps,  # (N,)
    }


def extract_running_speed(session):
    running_speed_dict = {}
    running_speed_df = session.running_speed
    if running_speed_df is not None:
        running_speed_df = running_speed_df[~running_speed_df.isnull().any(axis=1)]
        running_speed_times = running_speed_df.timestamps
        running_speed_dict.update(
            {
                "timestamps": running_speed_times.values,
                "running_speed": running_speed_df.speed
                .values.astype(np.float32)
                .reshape(-1, 1),  # continues values needs to be 2 dimensional
            }
        )
    return running_speed_dict

def get_stim_trial_splits(stim_dict, split_ratios=[0.7, 0.1, 0.2]):
    if stim_dict is None or len(stim_dict["timestamps"]) == 0:
        return {"train": None, "val": None, "test": None}
    import math

    train_boundary = math.floor(len(stim_dict["timestamps"]) * split_ratios[0])
    valid_boundary = math.floor(len(stim_dict["timestamps"]) * (split_ratios[0] + split_ratios[1]))
    test_boundary = math.floor(len(stim_dict["timestamps"]) * sum(split_ratios))
    
    stim_trials = np.vstack([stim_dict["start"], stim_dict["end"]]).T
    train_trials = stim_trials[:train_boundary - 1] 
    valid_trials = stim_trials[train_boundary: valid_boundary - 1] 
    test_trials = stim_trials[valid_boundary: test_boundary - 1] 
    
    return {"train": train_trials, "val": valid_trials, "test": test_trials}

def get_behavior_region(running_speed_dict, pupil_dict=None, gaze_dict=None):
    # extract session start and end times
    session_start = min(
        running_speed_dict["timestamps"].min() if running_speed_dict is not None else np.inf,
        pupil_dict["timestamps"].min() if pupil_dict is not None else np.inf,
    )
    session_end = max(
        running_speed_dict["timestamps"].max() if running_speed_dict is not None else 0,
        pupil_dict["timestamps"].max() if pupil_dict is not None else 0,
    )
    assert (
        session_start < session_end
    ), "Atleast one of running_speed, pupil or gaze data must be present."
    return session_start, session_end


def sample_free_behavior_splits(
    start, 
    end, 
    length=1, 
    sample_frac=0.7,
):

    sampled_begs = np.arange(start, end-length, length)
    sampled_ends = np.arange(start+length, end, length)
    all_chunks = np.c_[sampled_begs, sampled_ends]

    num_samples = int(len(all_chunks) * sample_frac)

    sampled_ids = np.random.choice(range(len(all_chunks)), num_samples, replace=False)
    sampled_chunks = all_chunks[sampled_ids]

    num_chunk = len(sampled_chunks)
    train_split = int(num_chunk * 0.7)
    val_split = int(num_chunk * 0.1)
    test_split = num_chunk - train_split - val_split 

    train_chunks = sampled_chunks[:train_split]
    val_chunks = sampled_chunks[train_split:train_split+val_split]
    test_chunks = sampled_chunks[train_split+val_split:]

    return {
        "train": np.array(sorted(train_chunks, key=lambda x: x[0])),
        "val": np.array(sorted(val_chunks, key=lambda x: x[0])),
        "test": np.array(sorted(test_chunks, key=lambda x: x[0])),
    }


def main():

    set_seed(42)

    # Use argparse to extract two arguments from the command line:
    # input_dir and output_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--session_id", type=int)
    args = parser.parse_args()

    input_dir = f"{args.data_dir}/raw"
    output_dir = f"{args.data_dir}/processed"

    # get the project cache from the warehouse
    manifest_path = os.path.join(input_dir, "manifest.json")
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=args.data_dir)
    # get sessions
    sessions = cache.get_ecephys_session_table()

    for session_id, row in tqdm(sessions.iterrows()):

        if session_id != args.session_id:
            continue
            
        # load nwb file through the allen sdk
        session_data = cache.get_ecephys_session(session_id)

        logging.info(f"Processing session: {session_id}")

        spikes, units = extract_spikes(session_data, session_id)

        # extract behavior and stimuli data
        # using dedicated extract_* helpers into a dictionary
        supervision_dict = {
            "running_speed": extract_running_speed(session_data),
            "rewards": extract_rewards(session_data),
        }
        
        # split each stimuli/behavior and combine them
        # using dedicated get_*_splits helpers into a dictionary
        stimuli_splits_by_key = {
            "rewards": get_stim_trial_splits(supervision_dict.get("rewards", None)),
        }

        behavior_start, behavior_end = get_behavior_region(
            supervision_dict.get("running_speed", None),
        )
        free_behavior_splits = sample_free_behavior_splits(behavior_start, behavior_end)

        session_dict = {"data": {}, "splits": {}}

        session_dict["data"] = {
            "spikes": spikes,
            "units": units,
            **supervision_dict,
        }

        session_dict["splits"] = {
            "free_behavior_splits": free_behavior_splits,
            **stimuli_splits_by_key,
        }

        with open(f"{output_dir}/{session_id}.pkl", "wb") as f:
            pickle.dump(session_dict, f)

        logging.info(f"Saved to disk session: {session_id}")

if __name__ == "__main__":
    main()
