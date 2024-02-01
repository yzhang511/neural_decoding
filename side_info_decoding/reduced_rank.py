import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from side_info_decoding.utils import sliding_window_over_trials


class Full_Rank_Model(nn.Module):
    """
    full rank model.
    """
    def __init__(
        self, 
        n_units, 
        n_t_bins, 
        half_window_size
    ):
        super(Full_Rank_Model, self).__init__()
        self.n_units = n_units
        self.n_t_bins = n_t_bins
        self.window_size = 2*half_window_size+1
        self.Beta = nn.Parameter(torch.randn(self.n_units, self.n_t_bins, self.window_size))
        self.intercept = nn.Parameter(torch.randn((1,)))
        self.sigmoid = nn.Sigmoid()
        self.task_type = "single_task"
        self.model_type = "full_rank"

    def forward(self, X):
        n_trials = len(X)
        out = torch.einsum("ctd,kctd->k", self.Beta, X)
        out += self.intercept * torch.ones(n_trials)
        out = self.sigmoid(out)
        return out, self.Beta


class Reduced_Rank_Model(Full_Rank_Model):
    """
    reduced rank model.
    """
    def __init__(
        self, 
        n_units, 
        n_t_bins, 
        rank, 
        half_window_size,
    ):
        super().__init__(
            n_units, 
            n_t_bins, 
            half_window_size,
        )
        self.rank = rank
        self.U = nn.Parameter(torch.randn(self.n_units, self.rank))
        self.V = nn.Parameter(torch.randn(self.rank, self.n_t_bins, self.window_size))
        self.task_type = "single_task"
        self.model_type = "reduced_rank"
        self.Beta = None

    def forward(self, X):
        n_trials = len(X)
        self.reduced_Beta = torch.einsum("cr,rtd->ctd", self.U, self.V)
        out = torch.einsum("ctd,kctd->k", self.reduced_Beta, X)
        out += self.intercept * torch.ones(n_trials)
        out = self.sigmoid(out)
        return out, self.U, self.V
    
# TO DO: check if this model makes sense; if not, remove it    
class Multi_Task_Full_Rank_Model(nn.Module):
    """
    multi-task version of full-rank models using soft sharing (l2 regularization).
    ref: https://avivnavon.github.io/blog/parameter-sharing-in-deep-learning/
    """
    def __init__(
        self, 
        n_tasks,
        n_units, 
        n_t_bins, 
        half_window_size,
        soft_loss_penalty=1e-1
    ):
        super(Multi_Task_Full_Rank_Model, self).__init__()
        self.n_tasks = n_tasks
        self.n_units = n_units
        self.n_t_bins = n_t_bins
        self.half_window_size = half_window_size
        self.window_size = 2*half_window_size+1
        self.soft_loss_penalty = soft_loss_penalty
        self.Betas = nn.ParameterList(
            [nn.Parameter(torch.randn(self.n_units[i], self.n_t_bins, self.window_size)) for i in range(self.n_tasks)]
        )
        self.intercepts = nn.ParameterList(
            [nn.Parameter(torch.randn(1,)) for i in range(self.n_tasks)]
        )
        self.sigmoid = nn.Sigmoid()
        self.task_type = "multi_task"
        self.model_type = "full_rank"

    def forward(self, X_lst):
        out_lst = []
        for task_idx in range(self.n_tasks):
            X = X_lst[task_idx]
            n_trials, n_units, n_t_bins, _ = X.shape
            out = torch.einsum("ctd,kctd->k", self.Betas[task_idx], X)
            out += self.intercepts[task_idx] * torch.ones(n_trials)
            out = self.sigmoid(out)
            out_lst.append(out)
        return out_lst, self.Betas
    
    def soft_loss(self):
        soft_sharing_loss = torch.zeros(self.n_t_bins)
        for t_bin_idx in range(self.n_t_bins):
            soft_sharing_loss[t_bin_idx] = self.soft_loss_penalty * torch.norm(
                torch.vstack([self.Betas[task_idx][:,t_bin_idx] for task_idx in range(self.n_tasks)]), p='fro'
            )
        return soft_sharing_loss.mean()
    
    
class Multi_Task_Reduced_Rank_Model(nn.Module):
    "multi-task version of reduced-rank models."
    def __init__(
        self, 
        n_tasks,
        n_units, 
        n_t_bins, 
        rank, 
        half_window_size,
    ):
        super(Multi_Task_Reduced_Rank_Model, self).__init__()
        self.n_tasks = n_tasks
        self.n_units = n_units
        self.n_t_bins = n_t_bins
        self.rank = rank
        self.half_window_size = half_window_size
        self.window_size = 2*half_window_size+1
        self.Us = nn.ParameterList(
            [nn.Parameter(torch.randn(self.n_units[i], self.rank)) for i in range(self.n_tasks)]
        )
        self.V = nn.Parameter(
            torch.randn(self.rank, self.n_t_bins, self.window_size)
        ) 
        self.intercepts = nn.ParameterList(
            [nn.Parameter(torch.randn(1,)) for i in range(self.n_tasks)]
        )
        self.sigmoid = nn.Sigmoid()
        self.task_type = "multi_task"
        self.model_type = "reduced_rank"

    def forward(self, X_lst):
        out_lst = []
        for task_idx in range(self.n_tasks):
            X = X_lst[task_idx]
            n_trials, n_units, n_t_bins, _ = X.shape
            self.Beta = torch.einsum("cr,rtd->ctd", self.Us[task_idx], self.V)
            out = torch.einsum("ctd,kctd->k", self.Beta, X)
            out += self.intercepts[task_idx] * torch.ones(n_trials)
            out = self.sigmoid(out)
            out_lst.append(out)
        return out_lst, self.Us, self.V
    
    
def train_single_task(
    model,
    train_dataset,
    test_dataset,
    loss_function=nn.BCELoss(),
    learning_rate=1e-3,
    weight_decay=1e-3,
    n_epochs=5000,
):
    """
    train reduced rank model.
    """
    train_X, train_Y = train_dataset
    test_X, test_Y = test_dataset
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    print_every_epochs = n_epochs//10

    # model training
    train_losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        if model.model_type == "reduced_rank":
            train_prob, _, _ = model(train_X)
        else:
            train_prob, _ = model(train_X)
        loss = loss_function(train_prob, train_Y)

        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.detach().numpy())
        if (epoch + 1) % print_every_epochs == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item()}")

    return model, train_losses


def train_multi_task(
    model,
    train_dataset,
    test_dataset,
    loss_function=nn.BCELoss(),
    learning_rate=1e-3,
    weight_decay=1e-3,
    n_epochs=5000,
):
    """
    train multi-task reduced rank model.
    """
    train_X, train_Y = train_dataset
    test_X, test_Y = test_dataset
    n_tasks = len(train_X)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    print_every_epochs = n_epochs//10

    # model training
    train_losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        if model.model_type == "reduced_rank":
            train_prob, _, _ = model(train_X)
        else:
            train_prob, _ = model(train_X)
            
        losses = torch.zeros((n_tasks,))
        for task_idx in range(n_tasks):
            losses[task_idx] = loss_function(train_prob[task_idx], train_Y[task_idx])
        loss = losses.mean()
        
        if model.model_type == "full_rank":
            loss += model.soft_loss()

        loss.backward()
        optimizer.step()
        
        train_losses.append(losses.detach().numpy())
        if (epoch + 1) % print_every_epochs == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item()}")

    return model, train_losses
    
    
    
def model_eval(
    model, 
    train_dataset,
    test_dataset,
    behavior="choice"
):
    
    train_X, train_Y = train_dataset
    test_X, test_Y = test_dataset
    n_tasks = len(train_X)
    
    with torch.no_grad():
        
        if model.task_type == "single_task":
            
            if model.model_type == "full_rank":
                train_prob, train_Beta = model(train_X)
                test_prob, test_Beta = model(test_X)
                test_Beta = test_Beta.detach().numpy()
            else:
                train_prob, train_U, train_V = model(train_X)
                test_prob, test_U, test_V = model(test_X)
                test_U = test_U.detach().numpy()
                test_V = test_V.detach().numpy()
            
            if behavior=="choice":
                train_pred = np.array(train_prob >= 0.5) * 1.
                test_pred = np.array(test_prob >= 0.5) * 1.
                train_acc = accuracy_score(train_Y, train_pred)
                test_acc = accuracy_score(test_Y, test_pred)
                try:
                    train_auc = roc_auc_score(train_Y, train_prob)
                    test_auc = roc_auc_score(test_Y, test_prob)
                except:
                    train_auc = np.nan
                    test_auc = np.nan
                print(f"train accuracy: {train_acc:.3f} auc: {test_acc:.3f}")
                print(f"test accuracy: {test_acc:.3f} auc: {test_auc:.3f}")
                test_metrics = [test_acc, test_auc]
            elif behavior=="prior":
                train_r2 = r2_score(train_Y, train_prob)
                test_r2 = r2_score(test_Y, test_prob)
                train_corr = pearsonr(train_Y, train_prob)[0]
                test_corr = pearsonr(test_Y, test_prob)[0]
                print(f"train r2: {train_r2:.3f} corr: {train_corr:.3f}")
                print(f"test r2: {test_r2:.3f} corr: {test_corr:.3f}")
                test_metrics = [test_r2, test_corr]
                
        elif model.task_type == "multi_task":
            
            if model.model_type == "full_rank":
                train_prob, train_Beta = model(train_X)
                test_prob, test_Beta = model(test_X)
                test_Beta = [Beta.detach().numpy() for Beta in test_Beta]
            else:
                train_prob, train_U, train_V = model(train_X)
                test_prob, test_U, test_V = model(test_X)
                test_U = [U.detach().numpy() for U in test_U]
                test_V = test_V.detach().numpy()
            
            test_metrics = []
            for task_idx in range(n_tasks):
                train_pred = np.array(train_prob[task_idx] >= 0.5) * 1.
                test_pred = np.array(test_prob[task_idx] >= 0.5) * 1.
                if behavior=="choice":
                    train_acc = accuracy_score(train_Y[task_idx], train_pred)
                    test_acc = accuracy_score(test_Y[task_idx], test_pred)
                    try:
                        train_auc = roc_auc_score(train_Y[task_idx], train_prob[task_idx])
                        test_auc = roc_auc_score(test_Y[task_idx], test_prob[task_idx])
                    except:
                        train_auc = np.nan
                        test_auc = np.nan
                    print(f"task {task_idx} train accuracy: {train_acc:.3f} auc: {train_auc:.3f}")
                    print(f"task {task_idx} test accuracy: {test_acc:.3f} auc: {test_auc:.3f}")
                    test_metrics.append([test_acc, test_auc])
                elif behavior=="prior":
                    train_r2 = r2_score(train_Y[task_idx], train_prob[task_idx])
                    test_r2 = r2_score(test_Y[task_idx], test_prob[task_idx])
                    train_corr = pearsonr(train_Y[task_idx], train_prob[task_idx])[0]
                    test_corr = pearsonr(test_Y[task_idx], test_prob[task_idx])[0]
                    print(f"task {task_idx} train r2: {train_r2:.3f} corr: {train_corr:.3f}")
                    print(f"task {task_idx} test r2: {test_r2:.3f} corr: {test_corr:.3f}")
                    test_metrics.append([test_r2, test_corr])
                    
        if model.model_type == "full_rank":  
            return test_Beta, test_metrics, test_prob
        else:
            return test_U, test_V, test_metrics, test_prob
    