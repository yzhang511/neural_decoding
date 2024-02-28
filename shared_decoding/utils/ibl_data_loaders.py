import numpy as np
from pathlib import Path
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

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


class SingleSessionDataset(Dataset):
    def __init__(self, data_dir, eid, beh_name, device):
        file_path = Path(data_dir)/f'{eid}.npz'
        self.data = np.load(file_path, allow_pickle=True)
        self.spike_data = self.data['spike_data']
        self.behavior = self.data[beh_name]
        self.n_t_steps = self.spike_data.shape[1]
        self.n_units = self.spike_data.shape[2]
        print(f"spike data shape: {self.spike_data.shape}")
        print(f"behavior data shape: {self.behavior.shape}")

        # scaling spike data only on the train set                                                                                               
        self.spike_data, self.means, self.stds = standardize_spike_data(self.spike_data)

        # scaling behavior only on the train set
        self.scaler = preprocessing.StandardScaler().fit(self.behavior)
        self.behavior = self.scaler.transform(self.behavior)

        self.spike_data = to_tensor(self.spike_data, device).double()
        self.behavior = to_tensor(self.behavior, device).double()

    def __len__(self):
        return len(self.spike_data)

    def __getitem__(self, trial_idx):
        return self.spike_data[trial_idx], self.behavior[trial_idx]

    
class SingleSessionDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config['data_dir']
        self.eid = config['eid']
        self.beh_name = config['target']
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.n_workers = config['n_workers']

    def setup(self, stage=None):
        session_dataset = SingleSessionDataset(self.data_dir, self.eid, self.beh_name, self.device)
        self.config.update({'n_units': session_dataset.n_units, 'n_t_steps': session_dataset.n_t_steps})
    
        data_len = len(session_dataset)
        train_len, val_len = int(0.8*data_len), int(0.1*data_len)
        test_len = data_len - train_len - val_len

        self.train, self.val, self.test = torch.utils.data.random_split(
            session_dataset, [train_len, val_len, test_len]
        )

    def train_dataloader(self):
        if self.device.type == 'cuda':
            # setting num_workers > 0 triggers errors so leave it as it is for now
            data_loader = DataLoader(
              self.train, batch_size=self.batch_size, shuffle=True, #num_workers=self.n_workers, pin_memory=True
            )
        else:
            data_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        return data_loader

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, drop_last=True)

    