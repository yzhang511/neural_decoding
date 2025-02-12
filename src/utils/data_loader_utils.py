"""Data loaders for single/multi-session models."""
import numpy as np
from pathlib import Path
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch import LightningDataModule
import datasets
from utils.dataset_utils import get_binned_spikes_from_sparse

seed = 42

# ---------
# Helpers
# ---------

def to_tensor(x, device):
    return torch.tensor(x).to(device)

def standardize_spike_data(spike_data, means=None, stds=None):
    
    K, T, N = spike_data.shape
    if (means is None) and (stds == None):
        means, stds = np.empty((T, N)), np.empty((T, N))

    std_spike_data = spike_data.reshape((K, -1))
    std_spike_data[np.isnan(std_spike_data)] = 0
    for t in range(T):
        mean = np.mean(std_spike_data[:, t*N:(t+1)*N])
        std = np.std(std_spike_data[:, t*N:(t+1)*N])
        std_spike_data[:, t*N:(t+1)*N] -= mean
        if std != 0:
            std_spike_data[:, t*N:(t+1)*N] /= std
        means[t], stds[t] = mean, std
    std_spike_data = std_spike_data.reshape(K, T, N)
    return std_spike_data, means, stds

def get_binned_spikes(dataset):
    spikes_sparse_data_list = dataset['spikes_sparse_data']
    spikes_sparse_indices_list = dataset['spikes_sparse_indices']
    spikes_sparse_indptr_list = dataset['spikes_sparse_indptr']
    spikes_sparse_shape_list = dataset['spikes_sparse_shape']
    
    binned_spikes  = get_binned_spikes_from_sparse(
        spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list
    )
    return binned_spikes.astype(float)

# ----------------------------
# Single-session data loaders
# ----------------------------

class SingleSessionDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        eid, 
        beh_name, 
        target, 
        device, 
        split="train", 
        region=None,
        load_local=True,
        huggingface_org="ibl-repro-ephys",
        standardize=False,
    ):
        """Load and preprocess single-session datasets.
            
        Args:
            data_dir: data path.
            eid: session ID.
            beh_name: behavior name to be loaded, e.g., 'choice', 'wheel-speed'.
            target:
                'cls': classification for discrete behavior.
                'reg': regression for continuous behavior.
            split: data partition; options = ['train', 'val', 'test'].
            region: region name to be loaded, e.g., 'LP', 'CA1'.
            load_local: whether load cached data locally or remotely from Hugging Face. 
        """
        if load_local:
            dataset = datasets.load_from_disk(Path(data_dir)/eid)
        else:
            dataset = datasets.load_dataset(f"{huggingface_org}/{eid}_aligned", cache_dir=data_dir)
        
        if split == "val":
            try:
                # if val exists, load pre-partitioned validation set
                self.spike_data = get_binned_spikes(dataset[split])
                self.behavior = np.array(dataset[split][beh_name])
            except:
                # if not, partition training data into train and val
                tmp = dataset[split].train_test_split(test_size=0.1, seed=seed)
                self.spike_data = get_binned_spikes(tmp["test"])
                self.behavior = np.array(tmp["test"][beh_name])
        else:
            self.spike_data = get_binned_spikes(dataset[split])
            self.behavior = np.array(dataset[split][beh_name])
        
        self.sessions = np.array([eid] * len(self.spike_data))
        self.n_trials, self.n_t_steps, self.n_units = self.spike_data.shape
        self.neuron_regions = np.array(dataset[split]["cluster_regions"])[0]

        for re_idx, re_name in enumerate(self.neuron_regions):
            if ("PO" in re_name) or ("PoT" in re_name) or ("POL" in re_name):
                self.neuron_regions[re_idx] = "PO"
            elif "DG" in re_name:
                self.neuron_regions[re_idx] = "DG"
            elif ("VISa" in re_name) or ("VISam" in re_name):
                self.neuron_regions[re_idx] = "VISa"

        if region and region != "all":
            neuron_idxs = np.argwhere(self.neuron_regions == region).flatten()
            self.spike_data = self.spike_data[..., neuron_idxs]
            self.regions = np.array([region] * len(self.spike_data))
        else:
            self.regions = np.array(["all"] * len(self.spike_data))

        if target == "clf":
            enc = OneHotEncoder(handle_unknown="ignore")
            self.behavior = enc.fit_transform(self.behavior.reshape(-1, 1)).toarray()
        elif target == "reg":
            pass
            # self.scaler = preprocessing.StandardScaler().fit(self.behavior)
            # self.behavior = self.scaler.transform(self.behavior) 

        if np.isnan(self.behavior).sum() != 0:
            self.behavior[np.isnan(self.behavior)] = np.nanmean(self.behavior)
            print(f"{beh_name} in session {eid} contains NaNs; interpolate with trial-average.")

        self.spike_data = to_tensor(self.spike_data, device).double()
        self.behavior = to_tensor(self.behavior, device).double()
  
    def __len__(self):
        return self.n_trials

    def __getitem__(self, trial_idx):
        return (
            self.spike_data[trial_idx], self.behavior[trial_idx], 
            self.regions[trial_idx], self.sessions[trial_idx]
        )

    
class SingleSessionDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config["dirs"]["data_dir"]
        self.eid = config["eid"]
        self.beh_name = config["target"]
        self.target = config["model"]["target"]
        self.region = config["region"]
        self.device = config["training"]["device"]
        self.load_local = config["training"]["load_local"]
        self.batch_size = config["training"]["batch_size"]
        self.n_workers = config["data"]["num_workers"]

    def update_config(self):
        self.val = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.target, 
            self.device, "val", self.region, self.load_local
        )
        self.config.update({
            "n_units": self.val.n_units, 
            "n_t_steps": self.val.n_t_steps,
            "eid": self.eid, 
            "region": self.region
        })

    def setup(self, stage=None):
        """Call this function to load and preprocess data."""
        self.train = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.target, 
            self.device, "train", self.region, self.load_local
        )
        self.val = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.target, 
            self.device, "val", self.region, self.load_local
        )
        self.test = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.target, 
            self.device, "test", self.region, self.load_local
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=False)


# ---------------------------
# Multi-session data loaders
# ---------------------------

class MultiSessionDataModule(LightningDataModule):
    def __init__(self, eids, configs):
        """Load and preprocess multi-session datasets.
            
        Args:
            eids: a list of session IDs.
            configs: a list of data configs for each session.
        """
        super().__init__()
        self.eids = eids
        self.configs = configs
        self.batch_size = configs[0]['training']['batch_size']

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
                DataLoader(dm.val, batch_size = self.batch_size, shuffle=False, drop_last=False)
            )
            self.test.append(
                DataLoader(dm.test, batch_size = self.batch_size, shuffle=False, drop_last=False)
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
    def __init__(self, eids, configs):
        """Load and preprocess multi-session datasets.
            
        Args:
            eids: a list of session IDs.
            configs: a list of data configs for each session-region combination.
            query_region: a list of brain regions to decode from.
        """
        super().__init__()
        self.eids = eids
        self.configs = configs
        self.batch_size = configs[0]['training']['batch_size']
        self.query_region = configs[0]['query_region']

    def list_regions(self):
        """Call this function to list all available brain regions from the input sessions."""
        self.all_regions, self.regions_dict = [], {}
        for idx, eid in enumerate(self.eids):
            dm = SingleSessionDataModule(self.configs[idx])
            dm.setup()
            unique_regions = [roi for roi in np.unique(dm.train.neuron_regions) if roi not in ['root', 'void']]
            self.regions_dict[eid] = unique_regions
            self.all_regions.extend(unique_regions)
        self.all_regions = list(np.unique(self.all_regions))
        
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
                DataLoader(dm.val, batch_size = self.batch_size, shuffle=False, drop_last=False)
            )
            self.test.append(
                DataLoader(dm.test, batch_size = self.batch_size, shuffle=False, drop_last=False)
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
        
    
