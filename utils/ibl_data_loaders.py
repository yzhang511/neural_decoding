import torch
from torch.utils.data import Dataset, DataLoader

class SingleSessionDataset(Dataset):
    def __init__(self, eid, beh_name):
        file_path = Path('./data/')/f'{eid}.npz'
        self.data = np.load(file_path, allow_pickle=True)
        self.spike_data = torch.tensor(self.data['spike_data'])
        self.behavior = torch.tensor(self.data[beh_name])
    
    def __len__(self):
        return len(self.spike_data)
    
    def __getitem__(self, trial_idx):
        return self.spike_data[trial_idx],self.behavior[trial_idx]