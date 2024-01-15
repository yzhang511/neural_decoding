"""get temporal basis per brain region via command line."""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import random
import pandas as pd
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn
from scipy.linalg import svd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

from one.api import ONE
from ibllib.atlas import AllenAtlas

from side_info_decoding.utils import (
    set_seed, 
    load_data_from_pids, 
    sliding_window_over_trials
)
from side_info_decoding.reduced_rank import (
    Multi_Task_Reduced_Rank_Model, 
    train_multi_task, 
    model_eval
)

if __name__ == "__main__":
    
    # -- args
    ap = argparse.ArgumentParser()
    g = ap.add_argument_group("Data Input/Output")
    g.add_argument("--roi_idx", type=int)
    g.add_argument("--save_path", type=str)
    g.add_argument("--n_sess", default=20, type=int)
    args = ap.parse_args()

    seed = 666
    set_seed(seed)
    
    R = 2 
    d = 0 
    n_epochs = 7000            
    n_folds = 5
    
    ba = AllenAtlas()
    roi = ba.regions.acronym[args.roi_idx]
    print(f"get temporal basis for {args.roi_idx}-th region: {roi}")
    
    # bwm_session_file = Path(args.save_path)/"bwm_cache_sessions.pqt"
    bwm_session_file = "/mnt/3TB/yizi/decode-paper-brain-wide-map/decoding/bwm_cache_sessions.pqt"
    bwm_df = pd.read_parquet(bwm_session_file)
    
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", mode='remote')
    pids_per_region = one.search_insertions(atlas_acronym=[roi], query_type='remote')
    print(f"{roi}: {len(pids_per_region)} PIDs")

    pids = list(pids_per_region)[:args.n_sess]

    try:
        results = {}

        # load data
        X_dict, Y_dict = load_data_from_pids(
            pids,
            brain_region=roi.lower(),
            behavior="choice",
            data_type="all_ks",
            n_t_bins = 40,
            t_before=0.5,
            t_after=1.5,
            align_time_type='stimOn_times',
        )

        _, contrast_dict = load_data_from_pids(
            pids,
            brain_region=roi.lower(),
            behavior="contrast",
            data_type="good_ks",
            n_t_bins = 40,
            t_before=0.5,
            t_after=1.5,
            align_time_type='stimOn_times',
        )

        loaded_pids = list(X_dict.keys())

        contrast_level_dict = {}
        filter_trials_dict = {}
        for pid in loaded_pids:
            contrast_dict[pid] = np.nan_to_num(contrast_dict[pid], 0)
            contrast_level_dict[pid] = contrast_dict[pid].sum(1)
            filter_trials_dict[pid] = {}
            for level in np.unique(contrast_level_dict[pid]):
                filter_trials_dict[pid].update({level: np.argwhere(contrast_level_dict[pid] == level).flatten()})
            for val in np.unique(Y_dict[pid]):
                if val == 0:
                    direc = "L"
                else:
                    direc = "R"
                filter_trials_dict[pid].update({direc: np.argwhere(Y_dict[pid] == val).flatten()})

        # get temporal basis from all trials
        train_pids, n_units = [], []
        train_X_dict, train_Y_dict = {}, {}
        for pid in loaded_pids:
            X, Y = X_dict[pid], Y_dict[pid]
            K, C, T = X.shape
            if C < 10:
                continue
            train_pids.append(pid)
            n_units.append(C)
            X = sliding_window_over_trials(X, half_window_size=d)
            Y = sliding_window_over_trials(Y, half_window_size=d)
            X, Y = torch.tensor(X), torch.tensor(Y)
            train_X_dict.update({pid: X})
            train_Y_dict.update({pid: Y})

        train_X_lst = [train_X_dict[pid] for pid in train_pids]
        train_Y_lst = [train_Y_dict[pid] for pid in train_pids]

        multi_task_rrm = Multi_Task_Reduced_Rank_Model(
            n_tasks=len(train_pids),
            n_units=n_units, 
            n_t_bins=T, 
            rank=R, 
            half_window_size=d,
            init_Us = None,
            init_V = None,
        )

        rrm, train_losses = train_multi_task(
            model=multi_task_rrm,
            train_dataset=(train_X_lst, train_Y_lst),
            test_dataset=(train_X_lst, train_Y_lst),
            loss_function=torch.nn.BCELoss(),
            learning_rate=1e-3,
            weight_decay=1e-1,
            n_epochs=n_epochs,
        )

        init_Us = np.array([multi_task_rrm.Us[pid_idx].detach().numpy() for pid_idx in range(len(train_pids))])
        init_V = multi_task_rrm.V.detach().numpy()
        Us, Vs = {}, {}
        for pid_idx, pid in enumerate(train_pids):
            Us.update({pid: init_Us[pid_idx]})
            Vs.update({pid: init_V})

        svd_W, svd_U, svd_S, svd_VT, S_mul_VT, W_reduced = [], [], [], [], [], []
        for pid in train_pids:
            W = np.array(Us[pid]) @ np.array(Vs[pid]).squeeze()
            U, S, VT = svd(W)
            svd_W.append(W)
            svd_U.append(U[:, :R])
            svd_S.append(S[:R])
            svd_VT.append(VT[:R, :])
            if len(S) == 1:
                S_mul_VT.append(np.diag(S) @ VT[:1, :])
            else:
                S_mul_VT.append(np.diag(S[:R]) @ VT[:R, :])

        results.update({"all": S_mul_VT})

        for direc in ["L", "R"]:
            test_X_lst = [train_X_dict[pid][filter_trials_dict[pid][direc]] for pid in train_pids]
            test_Y_lst = [train_Y_dict[pid][filter_trials_dict[pid][direc]] for pid in train_pids]

            proj_lst = []
            for pid_idx, pid in enumerate(train_pids):
                proj = (test_X_lst[pid_idx].squeeze().numpy().transpose(0,-1,1) @ svd_U[pid_idx]) * S_mul_VT[pid_idx].T
                proj_lst.append(proj.mean(0))
            results.update({direc: np.array(proj_lst)})

        # get temporal basis from trials with different contrast levels
        for level in [.0625, .125, .25, 1.]:
            try:
                test_X_lst = [train_X_dict[pid][filter_trials_dict[pid][level]] for pid in train_pids]
                test_Y_lst = [train_Y_dict[pid][filter_trials_dict[pid][level]] for pid in train_pids]
            except:
                continue

            multi_task_rrm = Multi_Task_Reduced_Rank_Model(
                n_tasks=len(train_pids),
                n_units=n_units, 
                n_t_bins=T, 
                rank=R, 
                half_window_size=d,
            )

            rrm, train_losses = train_multi_task(
                model=multi_task_rrm,
                train_dataset=(test_X_lst, test_Y_lst),
                test_dataset=(test_X_lst, test_Y_lst),
                loss_function=torch.nn.BCELoss(),
                learning_rate=1e-3,
                weight_decay=1e-1,
                n_epochs=n_epochs,
            )

            test_U, test_V, _, _ = model_eval(
                multi_task_rrm, 
                train_dataset=(test_X_lst, test_Y_lst),
                test_dataset=(test_X_lst, test_Y_lst),
                behavior="choice"
            )

            Us, Vs = {}, {}
            for pid_idx, pid in enumerate(train_pids):
                Us.update({pid: test_U[pid_idx]})
                Vs.update({pid: test_V})

            svd_W, svd_U, svd_S, svd_VT, S_mul_VT, W_reduced = [], [], [], [], [], []
            for pid in train_pids:
                W = np.array(Us[pid]) @ np.array(Vs[pid]).squeeze()
                U, S, VT = svd(W)
                svd_W.append(W)
                svd_U.append(U[:, :R])
                svd_S.append(S[:R])
                svd_VT.append(VT[:R, :])
                if len(S) == 1:
                    S_mul_VT.append(np.diag(S) @ VT[:1, :])
                else:
                    S_mul_VT.append(np.diag(S[:R]) @ VT[:R, :])

            results.update({level: S_mul_VT})

        np.save(Path(args.save_path)/f"{roi}_timescale.npy", results)

        # run 5-fold xval to get decoding accuracy
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        train_pids, n_units = [], []
        train_X_dict, test_X_dict, train_Y_dict, test_Y_dict = {}, {}, {}, {}
        for pid in loaded_pids:
            X, Y = X_dict[pid], Y_dict[pid]
            K, C, T = X.shape
            if C < 10:
                continue
            train_pids.append(pid)
            n_units.append(C)
            X = sliding_window_over_trials(X, half_window_size=d)
            Y = sliding_window_over_trials(Y, half_window_size=d)
            X, Y = torch.tensor(X), torch.tensor(Y)
            train_X_dict.update({pid: [X[train] for train, _ in skf.split(X, Y)]})
            test_X_dict.update({pid: [X[test] for _, test in skf.split(X, Y)]})
            train_Y_dict.update({pid: [Y[train] for train, _ in skf.split(X, Y)]})
            test_Y_dict.update({pid: [Y[test] for _, test in skf.split(X, Y)]})

        metrics_per_fold = []
        for fold_idx in range(n_folds):

            print(f"{fold_idx+1}/{n_folds} folds:")
            train_X_lst = [train_X_dict[pid][fold_idx] for pid in train_pids]
            test_X_lst = [test_X_dict[pid][fold_idx] for pid in train_pids]
            train_Y_lst = [train_Y_dict[pid][fold_idx] for pid in train_pids]
            test_Y_lst = [test_Y_dict[pid][fold_idx] for pid in train_pids]

            # multi-task reduced rank model
            multi_task_rrm = Multi_Task_Reduced_Rank_Model(
                n_tasks=len(train_pids),
                n_units=n_units, 
                n_t_bins=T, 
                rank=R, 
                half_window_size=d
            )

            rrm, train_losses = train_multi_task(
                model=multi_task_rrm,
                train_dataset=(train_X_lst, train_Y_lst),
                test_dataset=(test_X_lst, test_Y_lst),
                loss_function=torch.nn.BCELoss(),
                learning_rate=1e-3,
                weight_decay=1e-1,
                n_epochs=n_epochs,
            )

            _, _, rrm_metrics, _ = model_eval(
                multi_task_rrm, 
                train_dataset=(train_X_lst, train_Y_lst),
                test_dataset=(test_X_lst, test_Y_lst),
                behavior="choice"
            )
            
            # single-session full rank model
            frm_metrics = []
            for pid_idx, pid in enumerate(train_pids):
                clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1, 1e2, 1e3]).fit(
                    train_X_lst[pid_idx].squeeze().numpy().reshape((len(train_X_lst[pid_idx]), -1)), 
                    train_Y_lst[pid_idx].numpy()
                )
                test_pred = clf.predict(
                    test_X_lst[pid_idx].squeeze().numpy().reshape((len(test_X_lst[pid_idx]), -1))
                )
                frm_metrics.append(
                    [accuracy_score(test_Y_lst[pid_idx].numpy(), test_pred), 
                     roc_auc_score(test_Y_lst[pid_idx].numpy(), test_pred)]
                )
            
            metrics_per_fold.append(np.c_[rrm_metrics, frm_metrics])

        metrics_dict = {}
        for pid_idx, pid in enumerate(train_pids):
            metrics_dict.update({pid: np.mean(metrics_per_fold, 0)[pid_idx]})
        np.save(Path(args.save_path)/f"{roi}_metrics.npy", metrics_dict)

    except Exception as e: 
        print(f"Error! {e}")
        print(f"Skipped brain region {roi}")

