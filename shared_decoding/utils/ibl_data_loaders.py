import numpy as np
from pathlib import Path
from sklearn import preprocessing
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from shared_decoding.utils.ibl_data_utils import seed_everything

seed = 0
seed_everything(seed)

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

def recon_from_pcs(y, pca, comp_idxs=[0]):
    if len(comp_idxs) == 1:
        recon_y = np.dot(pca.transform(y)[:,comp_idxs[0],None], pca.components_[None,comp_idxs[0],:])
    else:
        recon_y = np.dot(pca.transform(y)[:,comp_idxs], pca.components_[comp_idxs,:])
    recon_y += pca.mean_
    return recon_y    

class SingleSessionDataset(Dataset):
    def __init__(self, data_dir, eid, beh_name, device, imposter_id=None):
        if imposter_id == None:
            file_path = Path(data_dir)/f'{eid}.npz'
        else:
            file_path = Path(data_dir)/eid/f'imposter_{imposter_id}.npz'
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
        self.behavior[np.isnan(self.behavior)] = np.nanmean(self.behavior)
        print(self.behavior.shape)
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
        self.imposter_id = config['imposter_id']
        self.beh_name = config['target']
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.n_workers = config['n_workers']

    def setup(self, stage=None):
        session_dataset = SingleSessionDataset(
            self.data_dir, self.eid, self.beh_name, self.device, imposter_id=self.imposter_id
        )
        self.config.update({'n_units': session_dataset.n_units, 'n_t_steps': session_dataset.n_t_steps})
    
        data_len = len(session_dataset)
        train_len, val_len = int(0.8*data_len), int(0.1*data_len)
        test_len = data_len - train_len - val_len

        gen = torch.Generator()
        gen.manual_seed(seed)
        self.train, self.val, self.test = torch.utils.data.random_split(
            session_dataset, [train_len, val_len, test_len], generator=gen
        )

    def recon_from_pcs(self, comp_idxs=[0]):
        
        train_x, train_y = [], []
        for (x, y) in self.train:
            train_x.append(x.cpu())
            train_y.append(y.cpu())
        train_x = np.stack(train_x)
        train_y = np.stack(train_y)
        
        val_x, val_y = [], []
        for (x, y) in self.val:
            val_x.append(x.cpu())
            val_y.append(y.cpu())
        val_x = np.stack(val_x)
        val_y = np.stack(val_y)
        
        test_x, test_y = [], []
        for (x, y) in self.test:
            test_x.append(x.cpu())
            test_y.append(y.cpu())
        test_x = np.stack(test_x)
        test_y = np.stack(test_y)
        
        all_y = np.vstack([train_y, val_y, test_y])
    
        pca = PCA(n_components=self.config['n_t_steps'])
        pca.fit(all_y)

        _train_y = recon_from_pcs(train_y, pca, comp_idxs=comp_idxs)
        _val_y = recon_from_pcs(val_y, pca, comp_idxs=comp_idxs)
        _test_y = recon_from_pcs(test_y, pca, comp_idxs=comp_idxs)

        self.train = [
            (to_tensor(train_x[i], self.device), to_tensor(_train_y[i], self.device)) for i in range(len(train_x))
        ]
        self.val = [
            (to_tensor(val_x[i], self.device), to_tensor(_val_y[i], self.device)) for i in range(len(val_x))
        ]
        self.test = [
            (to_tensor(test_x[i], self.device), to_tensor(_test_y[i], self.device)) for i in range(len(test_x))
        ]  
        print('Reconstructed from PCs: ', comp_idxs)
        
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


class MultiSessionDataModule(LightningDataModule):
    def __init__(self, eids, configs, comp_idxs=[]):
        super().__init__()
        self.eids = eids
        self.configs = configs
        self.batch_size = configs[0]['batch_size']
        self.comp_idxs = comp_idxs

    def setup(self, stage=None):
        self.train, self.val, self.test = [], [], []
        for idx, eid in enumerate(self.eids):
            dm = SingleSessionDataModule(self.configs[idx])
            dm.setup()
            if len(self.comp_idxs) != 0:
                dm.recon_from_pcs(comp_idxs=self.comp_idxs)
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
    
    
