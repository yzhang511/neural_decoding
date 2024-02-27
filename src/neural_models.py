import torch
from lightning.pytorch import LightningModule, Trainer
from torchmetrics import R2Score
from torch.nn import functional as F


class BaselineDecoder(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.n_units = config['n_units']
        self.n_t_steps = config['n_t_steps']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.lr_factor = config['lr_factor']
        self.lr_patience = config['lr_patience']

        self.r2_score = R2Score(num_outputs=self.n_t_steps, multioutput='uniform_average')

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        return loss

    def validation_step(self, batch, batch_idx, print_str="val"):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.r2_score(pred, y)

        self.log(f"{print_str}_loss", loss, prog_bar=True)
        self.log(f"{print_str}_r2", self.r2_score, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # reuse the validation_step for testing
        return self.validation_step(batch, batch_idx, print_str='test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    
class ReducedRankDecoder(BaselineDecoder):
    def __init__(self, config):
        super().__init__(config)

        self.temporal_rank = config['temporal_rank']

        # define PyTorch model
        self.U = torch.nn.Parameter(torch.randn(self.n_units, self.temporal_rank))
        self.V = torch.nn.Parameter(torch.randn(self.temporal_rank, self.n_t_steps, self.n_t_steps))
        self.b = torch.nn.Parameter(torch.randn(self.n_t_steps,))
        self.double()

    def forward(self, x):
        self.B = torch.einsum('nr,rtd->ntd', self.U, self.V)
        pred = torch.einsum('ntd,ktn->kd', self.B, x)
        pred += self.b
        return pred
    
    
class LSTMDecoder(BaselineDecoder):
    def __init__(self, config):
        super().__init__(config)

        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']
        self.drop_out = config['drop_out']

        self.lstm = torch.nn.LSTM(
            input_size=self.n_units,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.drop_out,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(self.hidden_size, self.n_t_steps)
        self.double()

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        pred = self.linear(lstm_out[:,-1])
        return pred


def eval_model(train, test, model, model_type='reduced-rank', plot=False):
    
    train_x, train_y = [], []
    for (x, y) in train:
        train_x.append(x.cpu())
        train_y.append(y.cpu())
    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)

    test_x, test_y = [], []
    for (x, y) in test:
        test_x.append(x.cpu())
        test_y.append(y.cpu())
    test_x = torch.stack(test_x)
    test_y = np.stack(test_y)

    if model_type == 'reduced-rank':
        test_pred = model(test_x).detach().numpy()

    elif model_type == 'reduced-rank-latents':
        U = model.U.cpu().detach().numpy()
        V = model.V.cpu().detach().numpy()

        train_proj_on_U = np.einsum('ktc,cr->ktr', train_x, U)
        test_proj_on_U = np.einsum('ktc,cr->ktr', test_x, U)
        weighted_train_proj = np.einsum('kdr,rdt->ktr', train_proj_on_U, V)
        weighted_test_proj = np.einsum('kdr,rdt->ktr', test_proj_on_U, V)

        alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        regr = GridSearchCV(Ridge(), {'alpha': alphas})

        train_x, test_x = weighted_train_proj, weighted_test_proj
        regr.fit(train_x.reshape((train_x.shape[0], -1)), train_y)
        test_pred = regr.predict(test_x.reshape((test_x.shape[0], -1)))

    elif model_type in ['ridge', 'mlp']:
        train_x, test_x = train_x.numpy(), test_x.numpy()
        model.fit(train_x.reshape((train_x.shape[0], -1)), train_y)
        test_pred = model.predict(test_x.reshape((test_x.shape[0], -1)))

    elif model_type == 'lstm':
        test_pred = model(test_x).detach().numpy()

    else:
        raise NotImplementedError

    r2 = r2_score(test_y.flatten(), test_pred.flatten())

    if plot:
        plt.figure(figsize=(12, 2))
        plt.plot(test_y[:10].flatten(), c='k', linewidth=.5, label='target')
        plt.plot(test_pred[:10].flatten(), c='b', label='pred')
        plt.title(f"model: {model_type} R2: {r2: .3f}")
        plt.legend()
        plt.show()

    return r2, test_pred, test_y

