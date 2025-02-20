import os
import sys
import pickle
import logging

import uuid
import numpy as np
import multiprocessing
from tqdm import tqdm
from sklearn import preprocessing
from scipy.interpolate import interp1d
from iblutil.numerical import bincount2D

import torch

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader

from utils.registry import target_registry

REGRESSION = ["running_speed", "gaze", "pupil"]
CLASSIFICATION = ["gabors", "static_gratings", "drifting_gratings"]

logging.basicConfig(level=logging.INFO)

def to_tensor(x, device):
    return torch.tensor(x).to(device)

def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result

def bin_spike_count(
    times, 
    units, 
    start, 
    end, 
    binsize=0.02, 
    length=None,
    n_workers=1
):

    num_chunk = len(start)
    if length is None:
        min_length = min(end - start)
        if min_length < 1:
            length = round(min_length, 1)
        else:
            length = int(min_length)
    print(f"Min sequence length: {length}s")
    num_bin = int(np.ceil(length / binsize))

    unit_index = np.unique(units)
    unit_count = len(unit_index)

    @globalize
    def count_spike_per_chunk(chunk):
        chunk_id, t_beg, t_end = chunk
        mask = (times >= t_beg) & (times < t_end)
        times_curr = times[mask]
        clust_curr = units[mask]

        if len(times_curr) == 0:
            spike_per_chunk = np.zeros((unit_count, num_bin))
            tbin_ids = np.arange(unit_count)
        else:
            spike_per_chunk, tbin_ids, unit_ids = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end]
            )
            _, tbin_ids, _ = np.intersect1d(unit_index, unit_ids, return_indices=True)

        return spike_per_chunk[:,:num_bin], tbin_ids, chunk_id

    spike_count = np.zeros((num_chunk, unit_count, num_bin))

    chunks = list(zip(np.arange(num_chunk), start, end))

    if n_workers == 1:
        for chunk in tqdm(chunks):
            res = count_spike_per_chunk(chunk)
            spike_count[res[-1], res[1], :] += res[0]
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            with tqdm(total=num_chunk) as pbar:
                for res in pool.imap_unordered(count_spike_per_chunk, chunks):
                    pbar.update()
                    spike_count[res[-1], res[1], :] += res[0]
            pbar.close()
    return spike_count
    
def bin_target(
    times, 
    values, 
    start, 
    end, 
    binsize=0.02, 
    length=None,
    n_workers=1, 
):  
    num_chunk = len(start)
    if length is None:
        min_length = min(end - start)
        if min_length < 1:
            length = round(min_length, 1)
        else:
            length = int(min_length)
    num_bin = int(np.ceil(length / binsize))

    start_ids = np.searchsorted(times, start, side="right")
    end_ids = np.searchsorted(times, end, side="left")
    _times_list = [times[s_id:e_id] for s_id, e_id in zip(start_ids, end_ids)]
    _vals_list = [values[s_id:e_id] for s_id, e_id in zip(start_ids, end_ids)]

    times_list = [None for _ in range(len(_times_list))]
    vals_list = [None for _ in range(len(_times_list))]
    valid_mask = [None for _ in range(len(_times_list))]
    skip_reason = [None for _ in range(len(_times_list))]

    @globalize
    def interpolate_func(target):
        chunk_idx, target_time, target_val = target
        target_time, target_val = target_time.squeeze(), target_val.squeeze()

        is_valid, x_interp, y_interp = False, None, None
        
        if len(target_val) == 0:
            skip_reason = "target data not present"
            return chunk_idx, is_valid, x_interp, y_interp, skip_reason
        if np.sum(np.isnan(target_val)) > 0:
            skip_reason = "nans in target data"
            return chunk_idx, is_valid, x_interp, y_interp, skip_reason
        if np.isnan(start[chunk_idx]) or np.isnan(end[chunk_idx]):
            skip_reason = "bad interval data"
            return chunk_idx, is_valid, x_interp, y_interp, skip_reason
        if np.abs(start[chunk_idx] - target_time[0]) > binsize:
            skip_reason = "target data starts too late"
            return chunk_idx, is_valid, x_interp, y_interp, skip_reason
        if np.abs(end[chunk_idx] - target_time[-1]) > binsize:
            skip_reason = "target data ends too early"
            return chunk_idx, is_valid, x_interp, y_interp, skip_reason

        is_valid, skip_reason = True, None
        x_interp = np.linspace(start[chunk_idx] + binsize, end[chunk_idx], num_bin)

        if len(target_val.shape) > 1 and target_val.shape[1] > 1:
            y_interp_list = []
            for n in range(target_val.shape[1]):
                y_interp_list.append(
                    interp1d(
                        target_time, target_val[:,n], kind="linear", fill_value="extrapolate"
                    )(x_interp)
                )
            y_interp = np.hstack([y[:, None] for y in y_interp_list])
        else:
            y_interp = interp1d(
                target_time, target_val, kind="linear", fill_value="extrapolate"
            )(x_interp)
        return chunk_idx, is_valid, x_interp, y_interp, skip_reason

    with multiprocessing.Pool(processes=n_workers) as p:
        targets = list(zip(np.arange(num_chunk), _times_list, _vals_list))
        with tqdm(total=num_chunk) as pbar:
            for res in p.imap_unordered(interpolate_func, targets):
                pbar.update()
                valid_mask[res[0]] = res[1]
                times_list[res[0]] = res[2]
                vals_list[res[0]] = res[3]
                skip_reason[res[0]] = res[-1]
        pbar.close()
        p.close()

    times_out = np.array(times_list)[valid_mask]
    values_out = np.array(vals_list)[valid_mask]
    times_out = np.array([x.flatten() for x in times_out])
    values_out = np.array([x.flatten() for x in values_out])
    valid_mask = np.array(valid_mask)
    
    return times_out, values_out, valid_mask, skip_reason
  

class BaseDataset(Dataset):
    def __init__(
        self, 
        session_id,
        target, 
        data_dir="./processed", 
        split="train", 
        device="cpu",
        binsize=0.02,
        length=None,
        region="all",
        n_workers=1,
    ):
        """
        TO DO: Filter by brain region.
        """

        cached_dir = str(data_dir).replace("processed", "cached")
        cached_dir = f"{cached_dir}/{session_id}/{target}/{region}/{split}"
        if os.path.exists(cached_dir):
            print(f"Loading cached data from {cached_dir}")
            spike_count = np.load(f"{cached_dir}/spike_count.npy", allow_pickle=True)
            behavior = np.load(f"{cached_dir}/behavior.npy", allow_pickle=True)
            self.start = np.load(f"{cached_dir}/start.npy", allow_pickle=True)
            self.end = np.load(f"{cached_dir}/end.npy", allow_pickle=True)
        else:
            os.makedirs(cached_dir, exist_ok=True)

            with open(f"{data_dir}/{session_id}.pkl", "rb") as f:
                session_dict = pickle.load(f)

            # Load behavior first because we need to filter out invalid trials
            if target in REGRESSION:
                start, end = session_dict["splits"]["free_behavior_splits"][split].T
                _, behavior, valid_mask, _ = bin_target(
                    session_dict["data"][target]["timestamps"], 
                    session_dict["data"][target][target], 
                    start=start,
                    end=end,
                    binsize=binsize,
                    length=length,
                    n_workers=n_workers
                )
                scaler = preprocessing.MinMaxScaler().fit(behavior)
                behavior = scaler.transform(behavior) 
            elif target in CLASSIFICATION:
                start, end = session_dict["splits"][target][split].T
                target_name = "gabors_orientation" if target == "gabors" else "orientation"
                timestamps = session_dict["data"][target]["timestamps"]
                behavior = session_dict["data"][target][target_name]
                mask = np.any((timestamps[:, None] >= start) & (timestamps[:, None] < end), axis=1)
                behavior = behavior[mask]
                valid_mask = ~np.isnan(behavior)
            else:
                raise ValueError(f"Target {target} not supported.")

            spike_count = bin_spike_count(
                session_dict["data"]["spikes"], 
                session_dict["data"]["units"]["unit_index"], 
                start=start,
                end=end,
                binsize=binsize,
                length=length,
                n_workers=n_workers
            )

            # Filter out invalid trials
            spike_count = spike_count[valid_mask]
            self.start = start[valid_mask]
            self.end = end[valid_mask]

            np.save(f"{cached_dir}/spike_count.npy", spike_count)
            np.save(f"{cached_dir}/behavior.npy", behavior)
            np.save(f"{cached_dir}/start.npy", start)
            np.save(f"{cached_dir}/end.npy", end)
        
        self.num_trials, self.num_timesteps, self.num_units = spike_count.shape
        self.sessions = np.array([session_id] * self.num_trials)
        self.regions = np.array([region] * self.num_trials)

        self.spike_count = to_tensor(spike_count, device).double()
        self.behavior = to_tensor(behavior, device)
        self.behavior = self.behavior.double() if target in REGRESSION else self.behavior.long()

        if len(self.spike_count) == 0 or len(self.behavior) == 0:
            raise ValueError(f"No valid trials found for {session_id} {target} {split} set.")
        
    def __len__(self):
        return self.num_trials

    def __getitem__(self, trial_idx):
        return (
            self.spike_count[trial_idx], self.behavior[trial_idx], 
            self.regions[trial_idx], self.sessions[trial_idx]
        )

    
class SingleSessionDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config["dirs"]["data_dir"]
        self.session_id = config["session_id"]
        self.target = config["target"]
        self.region = config.get("region", "all")
        self.binsize = config.get("binsize", 0.01)
        self.length = config.get("length", None)
        self.device = config["training"].get("device", "cpu")
        self.batch_size = config.get("training", {}).get("batch_size", 16)
        self.n_workers = config.get("data", {}).get("num_workers", 1)

    def update_config(self):
        self.val = BaseDataset(
            self.session_id, self.target, self.data_dir, "val", 
            self.device, self.binsize, self.length, self.region, self.n_workers
        )
        self.config.update({
            "num_units": self.val.num_units, 
            "num_timesteps": self.val.num_timesteps,
            "session_id": self.session_id, 
            "region": self.region
        })

    def setup(self, stage=None):
        """Call this function to load and preprocess data."""
        self.train = BaseDataset(
            self.session_id, self.target, self.data_dir, "train", 
            self.device, self.binsize, self.length, self.region, self.n_workers
        )
        self.val = BaseDataset(
            self.session_id, self.target, self.data_dir, "val", 
            self.device, self.binsize, self.length, self.region, self.n_workers
        )
        self.test = BaseDataset(
            self.session_id, self.target, self.data_dir, "test", 
            self.device, self.binsize, self.length, self.region, self.n_workers
        )

    def train_dataloader(self):
        data_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        return data_loader

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=False)


class MultiSessionDataModule(LightningDataModule):
    def __init__(self, session_ids, configs):
        """Load and preprocess multi-session datasets.
            
        Args:
            session_ids: a list of session IDs.
            configs: a list of data configs for each session.
        """
        super().__init__()
        self.session_ids = session_ids
        self.configs = configs
        self.batch_size = configs[0].get("training", {}).get("batch_size", 16)

    def setup(self, stage=None):
        """Call this function to load and preprocess data."""
        self.train, self.val, self.test = [], [], []
        for config in self.configs:
            dm = SingleSessionDataModule(config)
            dm.setup()
            self.train.append(
                DataLoader(dm.train, batch_size = self.batch_size, shuffle=True)
            )
            self.val.append(
                DataLoader(dm.val, batch_size = self.batch_size, shuffle=False, drop_last=True)
            )
            self.test.append(
                DataLoader(dm.test, batch_size = self.batch_size, shuffle=False, drop_last=True)
            )

    def train_dataloader(self):
        data_loader = CombinedLoader(self.train, mode = "max_size_cycle")
        return data_loader

    def val_dataloader(self):
        data_loader = CombinedLoader(self.val)
        return data_loader

    def test_dataloader(self):
        data_loader = CombinedLoader(self.test)
        return data_loader


class MultiRegionDataModule(LightningDataModule):
    def __init__(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
        
    