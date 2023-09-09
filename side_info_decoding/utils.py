import numpy as np
import random
import torch
from density_decoding.utils.data_utils import IBLDataLoader

def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.set_default_dtype(torch.double)
    
    
def load_data_from_pids(
    pids,
    brain_region,
    behavior="choice",
    data_type="all_ks",
    n_t_bins=30,
    prior_path=None
):
    X_dict, Y_dict = {}, {}
    for pid_idx in range(len(pids)):
        pid = pids[pid_idx]
        ibl_data_loader = IBLDataLoader(
          pid,
          trial_length = 1.5,
          n_t_bins = n_t_bins,
          prior_path = prior_path
        )
        Y = ibl_data_loader.process_behaviors(behavior)
        Y_dict.update({pid: Y})
        if data_type == "all_ks":
            X = ibl_data_loader.load_all_sorted_units(brain_region)
        elif data_type == "good_ks":
            X = ibl_data_loader.load_good_sorted_units(brain_region)
        elif data_type == "thresholded":
            X = ibl_data_loader.load_thresholded_units(brain_region)
        else:
            raise TypeError("other neural data types are under development.")
        X_dict.update({pid: normalize_data(X)})
    return X_dict, Y_dict

    

def sliding_window_over_trials(data, half_window_size=0):
    """
    apply sliding window over either neural data or behavior data.
    """
    window_size = 2*half_window_size+1
    if len(data.shape) == 3:
        n_trials, n_units, n_t_bins = data.shape
        data_window = np.zeros((
            n_trials - 2*half_window_size, n_units, n_t_bins, window_size
        ))
        for k in range(n_trials - 2*half_window_size):
            tmp_window = np.zeros((n_units, n_t_bins, window_size))
            for d in range(window_size):
                  tmp_window[:,:,d] = data[k + d]
            data_window[k] = tmp_window
    else:
        n_trials = data.shape[0]
        max_len = n_trials - 2*half_window_size
        data_window = np.zeros((max_len, window_size))
        for k in range(max_len):
            data_window[k] = data[k:k+window_size]
        data_window = data_window[:,half_window_size]
    return data_window


def normalize_data(data):
    """
    normalize neural data for SGD.
    """
    n_trials, n_units, n_t_bins = data.shape

    data_norm = data.reshape(-1, n_units*n_t_bins)
    for t in range(n_t_bins):
        mean_per_trial = np.nanmean(data_norm[:,t*n_units:(t+1)*n_units])
        std_per_trial = np.nanstd(data_norm[:, t*n_units:(t+1)*n_units])
        data_norm[:,t*n_units:(t+1)*n_units] = (data_norm[:,t*n_units:(t+1)*n_units] - mean_per_trial) / std_per_trial
    data_norm[np.isnan(data_norm)] = np.nanmean(data_norm)  
    data_norm = data_norm.reshape(-1, n_units, n_t_bins)
    return data_norm

