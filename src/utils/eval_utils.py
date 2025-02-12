"""Evaluation for single/multi-session decoders."""
import os
import numpy as np
import statistics as st
import torch
from torch.nn.functional import softmax
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import gammaln
from sklearn.cluster import SpectralClustering
import datasets

# --------------------------
# Single-session evaluation
# --------------------------

def eval_model(
    train, test, model, 
    target='reg', 
    model_class='reduced_rank', 
    training_type='single-sess', 
):
    """Model evaluation backbone.
        
    Args:
        train: training data (torch.utils.data.Dataset). 
        test: test data (torch.utils.data.Dataset). 
        model: pretrained PyTorch model.
        target: 
            'cls': classification for discrete behavior.
            'reg': regression for continuous behavior.
        model_class: options = ['linear', 'reduced_rank', 'mlp', 'lstm'].
        training_type: options = ['single-sess', 'multi-sess'].
    """
    # Load data
    train_x, train_y = [], []
    for (x, y, region, eid) in train:
        train_x.append(x.cpu())
        train_y.append(y.cpu())

    test_x, test_y = [], []
    for (x, y, region, eid) in test:
        test_x.append(x.cpu())
        test_y.append(y.cpu())
        
    if training_type == 'multi-sess':
        train_x, train_y = torch.vstack(train_x), torch.vstack(train_y)
        test_x, test_y = torch.vstack(test_x), np.vstack(test_y)
    elif training_type == 'single-sess':
        train_x, train_y = torch.stack(train_x), torch.stack(train_y)
        test_x, test_y = torch.stack(test_x), np.stack(test_y)
    else:
        raise NotImplementedError

    # Model prediction
    if model_class == 'reduced_rank':
        if training_type == 'multi-sess':
            test_pred = model(test_x, eid[0], region[0])
        else:
            test_pred = model(test_x)
        if target == 'clf':
            test_pred = softmax(test_pred, dim=1).detach().numpy().argmax(1)
        elif target == 'reg':
            test_pred = test_pred.detach().numpy()
    elif model_class == 'linear':
        train_x, test_x = train_x.numpy(), test_x.numpy()
        if target == 'clf':
            model.fit(train_x.reshape((train_x.shape[0], -1)), train_y)
        elif target == 'reg':
            model.fit(train_x.reshape((train_x.shape[0], -1)), train_y)
        test_pred = model.predict(test_x.reshape((test_x.shape[0], -1)))
    elif model_class in ['mlp', 'lstm']:
        test_pred = model(test_x)
        if target == 'clf':
            test_pred = softmax(test_pred, dim=1).detach().numpy().argmax(1)
        elif target == 'reg':
            test_pred = test_pred.detach().numpy()
    else:
        raise NotImplementedError

    # Evaluation
    if target == 'reg':
        metric = r2_score(test_y.flatten(), test_pred.flatten())
    elif target == 'clf':
        metric = accuracy_score(test_y, test_pred)
    else:
        raise NotImplementedError
        
    return metric, test_pred, test_y


# --------------------------
# Multi-session evaluation
# --------------------------

def eval_multi_session_model(
    train_lst, 
    test_lst, 
    model, 
    target='reg',
    model_class='reduced_rank', 
    beh_name=None,
    save_path=None,
    data_dir=None,
    load_local=True,
    huggingface_org="neurofm123",
    configs=None,
):
    """Multi-session model evaluation.
        
    Args:
        train_lst: a list of training data (torch.utils.data.Dataset) from different sessions. 
        test_lst: a list of test data (torch.utils.data.Dataset) from different sessions. 
        model: pretrained PyTorch model.
        target: 
            'cls': classification for discrete behavior.
            'reg': regression for continuous behavior.
        model_class: options = ['reduced_rank'].
    """
    assert model_class=='reduced_rank', 'Other models do not support multi-session training yet. '
    
    metric_lst, chance_metric_lst, test_pred_lst, test_y_lst = [], [], [], []
    for idx, (train, test) in enumerate(zip(train_lst, test_lst)):
        eid = configs[idx]['eid']
        kwargs = {
            "eid": eid,
            "beh": beh_name,
            "region": "all",
            "save_path": save_path/region/eid,
            "data_dir": data_dir,
            "huggingface_org": huggingface_org,
            "load_local": load_local,
        }
        metric, chance_metric, test_pred, test_y = eval_model(
            train, test, model, 
            target=target, 
            model_class=model_class, 
            training_type='multi-sess',
            **kwargs
        )
        metric_lst.append(metric)
        chance_metric_lst.append(chance_metric)
        test_pred_lst.append(test_pred)
        test_y_lst.append(test_y)

    return metric_lst, chance_metric_lst, test_pred_lst, test_y_lst


# --------------------------
# Multi-region evaluation
# --------------------------

def eval_multi_region_model(
    train_lst, 
    test_lst, 
    model, 
    target='reg',
    model_class='reduced_rank',
    beh_name=None,
    save_path=None,
    data_dir=None,
    load_local=True,
    huggingface_org="neurofm123",
    all_regions=None,
    configs=None,
):
    """Multi-region model evaluation.
        
    Args:
        train_lst: a list of training data (torch.utils.data.Dataset) from different sessions. 
        test_lst: a list of test data (torch.utils.data.Dataset) from different sessions. 
        model: pretrained PyTorch model.
        target: 
            'cls': classification for discrete behavior.
            'reg': regression for continuous behavior.
        model_class: options = ['reduced_rank'].
        all_regions: a list of all available input regions.
        configs: a list of model configs for each session-region combination. 
    """
    assert model_class=='reduced_rank', 'Other models do not support multi-region training yet. '
    
    metric_dict, chance_metric_dict, test_pred_dict, test_y_dict = {}, {}, {}, {}
    for region in all_regions:
        metric_dict[region], chance_metric_dict[region] = {}, {}
        test_pred_dict[region], test_y_dict[region] = {}, {}
    
    model.eval()
    for idx, (train, test) in enumerate(zip(train_lst, test_lst)):
        eid = configs[idx]['eid']
        region = configs[idx]['region']
        kwargs = {
            "eid": eid,
            "beh": beh_name,
            "region": region,
            "save_path": save_path/region/eid,
            "data_dir": data_dir,
            "huggingface_org": huggingface_org,
            "load_local": load_local,
        }
        metric, chance_metric, test_pred, test_y = eval_model(
            train, test, model, 
            target=target, 
            model_class=model_class, 
            training_type='multi-sess',
            **kwargs
        )
        metric_dict[region][eid] = metric
        chance_metric_dict[region][eid] = chance_metric
        test_pred_dict[region][eid] = test_pred
        test_y_dict[region][eid] = test_y

    return metric_dict, chance_metric_dict, test_pred_dict, test_y_dict
    

# ----------------
# Plot functions
# ----------------

def viz_single_cell(X, y, y_pred, var_name2idx, var_tasklist, var_value2label, var_behlist,
                    subtract_psth="task", aligned_tbins=[], clusby='y_pred', neuron_idx='', neuron_region='', method='',
                    save_path='figs', save_plot=False):
    
    if save_plot:
        nrows = 8
        plt.figure(figsize=(8, 2 * nrows))
        axes_psth = [plt.subplot(nrows, len(var_tasklist), k + 1) for k in range(len(var_tasklist))]
        axes_single = [plt.subplot(nrows, 1, k) for k in range(2, 2 + 2 + len(var_behlist) + 2)]
    else:
        axes_psth = None
        axes_single = None


    ### plot psth
    r2_psth, r2_trial = plot_psth(X, y, y_pred,
                                  var_tasklist=var_tasklist,
                                  var_name2idx=var_name2idx,
                                  var_value2label=var_value2label,
                                  aligned_tbins=aligned_tbins,
                                  axes=axes_psth, legend=True, neuron_idx=neuron_idx, neuron_region=neuron_region,
                                  save_plot=save_plot)

    ### plot the psth-subtracted activity
    if save_plot:
        plot_single_trial_activity(X, y, y_pred,
                                   var_name2idx,
                                   var_behlist,
                                   var_tasklist, subtract_psth=subtract_psth,
                                   aligned_tbins=aligned_tbins,
                                   clusby=clusby,
                                   axes=axes_single)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if save_plot:
        plt.savefig(os.path.join(save_path, f"{neuron_region.replace('/', '-')}_{neuron_idx}_{r2_trial:.2f}_{method}.png"))
        plt.tight_layout()

    return r2_psth, r2_trial


def create_behave_list(batch, T=40):
    
    b_list = []
        
    choice = np.array(batch['choice'])
    choice = np.tile(np.reshape(choice, (choice.shape[0], 1)), (1, T))
    b_list.append(choice)
    
    reward = np.array(batch['reward'])
    reward = np.tile(np.reshape(reward, (reward.shape[0], 1)), (1, T))
    b_list.append(reward)
    
    stimside = np.array(batch['stimside'])
    stimside = np.tile(np.reshape(stimside, (stimside.shape[0], 1)), (1, T))
    b_list.append(stimside)
    
    behavior_set = np.stack(b_list, axis=-1)
    
    var_name2idx = {'stimside': [2], 'choice': [0], 'reward': [1], 'wheel': [3]}
    var_value2label = {'stimside': {(0.,): "right", (1.,): "left",},
                       'choice': {(-1.0,): "right", (1.0,): "left"},
                       'reward': {(0.,): "no reward", (1.,): "reward",}}
    var_tasklist = ['stimside', 'choice', 'reward']
    var_behlist = []
    
    return behavior_set, var_name2idx, var_tasklist, var_value2label, var_behlist


def plot_psth(X, y, y_pred, var_tasklist, var_name2idx, var_value2label,
              aligned_tbins=[],
              axes=None, legend=False, neuron_idx='', neuron_region='', save_plot=False):
    
    if save_plot:
        if axes is None:
            nrows = 1;
            ncols = len(var_tasklist)
            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))

        for ci, var in enumerate(var_tasklist):
            ax = axes[ci]
            psth_xy = compute_all_psth(X, y, var_name2idx[var])
            psth_pred_xy = compute_all_psth(X, y_pred, var_name2idx[var])
            
            for _i, _x in enumerate(psth_xy.keys()):
                
                psth = psth_xy[_x]
                psth_pred = psth_pred_xy[_x]
                ax.plot(psth,
                        color=plt.get_cmap('tab10')(_i),
                        linewidth=3, alpha=0.3, label=f"{var}: {tuple(_x)[0]:.2f}")
                ax.plot(psth_pred,
                        color=plt.get_cmap('tab10')(_i),
                        linestyle='--')
                ax.set_xlabel("Time bin")
                if ci == 0:
                    ax.set_ylabel("Neural activity")
                else:
                    ax.sharey(axes[0])
            _add_baseline(ax, aligned_tbins=aligned_tbins)
            if legend:
                ax.legend()
                ax.set_title(f"{var}")

    # compute PSTH for task_contingency
    idxs_psth = np.concatenate([var_name2idx[var] for var in var_tasklist])
    psth_xy = compute_all_psth(X, y, idxs_psth)
    psth_pred_xy = compute_all_psth(X, y_pred, idxs_psth)
    r2_psth = compute_R2_psth(psth_xy, psth_pred_xy, clip=False)
    r2_single_trial = compute_R2_main(y.reshape(-1, 1), y_pred.reshape(-1, 1), clip=False)[0]
    
    if save_plot:
        axes[0].set_ylabel(
            f'Neuron: #{neuron_idx[:4]} \n PSTH R2: {r2_psth:.2f} \n Avg_SingleTrial R2: {r2_single_trial:.2f}')

        for ax in axes:
            # ax.axis('off')
            ax.spines[['right', 'top']].set_visible(False)
            # ax.set_frame_on(False)
            # ax.tick_params(bottom=False, left=False)
        plt.tight_layout()

    return r2_psth, r2_single_trial


def plot_single_trial_activity(X, y, y_pred,
                               var_name2idx,
                               var_behlist,
                               var_tasklist, subtract_psth="task",
                               aligned_tbins=[],
                               n_clus=8, n_neighbors=5, n_pc=32, clusby='y_pred',
                               cmap='bwr', vmax_perc=90, vmin_perc=10,
                               axes=None):
    if axes is None:
        ncols = 1;
        nrows = 2 + len(var_behlist) + 1 + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3 * nrows))

    ### get the psth-subtracted y
    if subtract_psth is None:
        pass
    elif subtract_psth == "task":
        idxs_psth = np.concatenate([var_name2idx[var] for var in var_tasklist])
        psth_xy = compute_all_psth(X, y, idxs_psth)
        psth_pred_xy = compute_all_psth(X, y_pred, idxs_psth)
        y_psth = np.asarray(
            [psth_xy[tuple(x)] for x in X[:, 0, idxs_psth]])  # (K, T) predict the neural activity with psth
        y_predpsth = np.asarray(
            [psth_pred_xy[tuple(x)] for x in X[:, 0, idxs_psth]])  # (K, T) predict the neural activity with psth
        y = y - y_psth  # (K, T)
        y_pred = y_pred - y_predpsth  # (K, T)
    elif subtract_psth == "global":
        y_psth = np.mean(y, 0)
        y_predpsth = np.mean(y_pred, 0)
        y = y - y_psth  # (K, T)
        y_pred = y_pred - y_predpsth  # (K, T)
    else:
        assert False, "Unknown subtract_psth, has to be one of: task, global. \'\'"

    y_residual = (y_pred - y)  # (K, T), residuals of prediction
    idxs_behavior = np.concatenate(([var_name2idx[var] for var in var_behlist])) if len(var_behlist) > 0 else []
    X_behs = X[:, :, idxs_behavior]

    clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                    affinity='nearest_neighbors',
                                    assign_labels='discretize',
                                    random_state=0)
    if clusby == 'y_pred':
        clustering = clustering.fit(y_pred)
    elif clusby == 'y':
        clustering = clustering.fit(y)
    else:
        assert False, "invalid clusby"
    t_sort = np.argsort(clustering.labels_)

    for ri, (toshow, label, ax) in enumerate(zip([y, y_pred, X_behs, y_residual],
                                                 [f"obs. act. \n (subtract_psth={subtract_psth})",
                                                  f"pred. act. \n (subtract_psth={subtract_psth})",
                                                  var_behlist,
                                                  "residual act."],
                                                 [axes[0], axes[1], axes[2:-2], axes[-2]])):
        if ri <= 1:
            # plot obs./ predicted activity
            vmax = np.percentile(y_pred, vmax_perc)
            vmin = np.percentile(y_pred, vmin_perc)
            raster_plot(toshow[t_sort], vmax, vmin, True, label, ax,
                        cmap=cmap,
                        aligned_tbins=aligned_tbins)
        elif ri == 2:
            # plot behavior
            for bi in range(len(var_behlist)):
                ts_ = toshow[:, :, bi][t_sort]
                vmax = np.percentile(ts_, vmax_perc)
                vmin = np.percentile(ts_, vmin_perc)
                raster_plot(ts_, vmax, vmin, True, label[bi], ax[bi],
                            cmap=cmap,
                            aligned_tbins=aligned_tbins)
        elif ri == 3:
            # plot residual activity
            vmax = np.percentile(toshow, vmax_perc)
            vmin = np.percentile(toshow, vmin_perc)
            raster_plot(toshow[t_sort], vmax, vmin, True, label, ax,
                        cmap=cmap,
                        aligned_tbins=aligned_tbins)

    ### plot single-trial activity
    # re-arrange the trials
    clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                    affinity='nearest_neighbors',
                                    assign_labels='discretize',
                                    random_state=0).fit(y_residual)
    t_sort_rd = np.argsort(clustering.labels_)
    # model = Rastermap(n_clusters=n_clus, n_PCs=n_pc, locality=0.15, time_lag_window=15, grid_upsample=0,).fit(y_residual)
    # t_sort_rd = model.isort
    raster_plot(y_residual[t_sort_rd], np.percentile(y_residual, vmax_perc), np.percentile(y_residual, vmin_perc), True,
                "residual act. (re-clustered)", axes[-1])

    plt.tight_layout()


def _add_baseline(ax, aligned_tbins=[40]):
    for tbin in aligned_tbins:
        ax.axvline(x=tbin - 1, c='k', alpha=0.2)


def raster_plot(ts_, vmax, vmin, whether_cbar, ylabel, ax,
                cmap='bwr',
                aligned_tbins=[40]):
    N, T = ts_.shape
    im = ax.imshow(ts_, aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin)
    for tbin in aligned_tbins:
        ax.annotate('',
                    xy=(tbin - 1, N),
                    xytext=(tbin - 1, N + 10),
                    ha='center',
                    va='center',
                    arrowprops={'arrowstyle': '->', 'color': 'r'})
    if whether_cbar:
        cbar = plt.colorbar(im, pad=0.01, shrink=.6)
        cbar.ax.tick_params(rotation=90)
    if not (ylabel is None):
        ax.set_ylabel(f"{ylabel}" + f"\n(#trials={N})")
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
        pass
    else:
        ax.axis('off')


def compute_PSTH(X, y, axis, value):
    trials = np.all(X[:, 0, axis] == value, axis=-1)
    return y[trials].mean(0)


def compute_all_psth(X, y, idxs_psth):
    uni_vs = np.unique(X[:, 0, idxs_psth], axis=0)  # get all the unique task-conditions
    psth_vs = {};
    for v in uni_vs:
        # compute separately for true y and predicted y
        _psth = compute_PSTH(X, y,
                             axis=idxs_psth, value=v)  # (T)
        psth_vs[tuple(v)] = _psth
    return psth_vs


def compute_R2_psth(psth_xy, psth_pred_xy, clip=True):
    psth_xy_array = np.array([psth_xy[x] for x in psth_xy])
    psth_pred_xy_array = np.array([psth_pred_xy[x] for x in psth_xy])
    K, T = psth_xy_array.shape[:2]
    psth_xy_array = psth_xy_array.reshape((K * T, -1))
    psth_pred_xy_array = psth_pred_xy_array.reshape((K * T, -1))
    r2s = [r2_score(psth_xy_array[:, ni], psth_pred_xy_array[:, ni]) for ni in range(psth_xy_array.shape[1])]
    r2s = np.array(r2s)
    # # compute r2 along dim 0
    # r2s = [r2_score(psth_xy[x], psth_pred_xy[x], multioutput='raw_values') for x in psth_xy]
    if clip:
        r2s = np.clip(r2s, 0., 1.)
    # r2s = np.mean(r2s, 0)
    if len(r2s) == 1:
        r2s = r2s[0]
    return r2s


def compute_R2_main(y, y_pred, clip=True):
    """
    :y: (K, T, N) or (K*T, N)
    :y_pred: (K, T, N) or (K*T, N)
    """
    N = y.shape[-1]
    if len(y.shape) > 2:
        y = y.reshape((-1, N))
    if len(y_pred.shape) > 2:
        y_pred = y_pred.reshape((-1, N))
    r2s = np.asarray([r2_score(y[:, n].flatten(), y_pred[:, n].flatten()) for n in range(N)])
    if clip:
        return np.clip(r2s, 0., 1.)
    else:
        return r2s
        