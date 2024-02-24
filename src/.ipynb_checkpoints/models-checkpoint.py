import torch
from lightning.pytorch import LightningModule, Trainer
from torchmetrics import R2Score
from torch.nn import functional as F

class ReducedRankModel(LightningModule):
    def __init__(self, train, test, val, config):
        super().__init__()
        self.train_data = train
        self.val_data = val
        self.test_data = test
        self.temporal_rank = config['temporal_rank']
        self.n_units = config['n_units']
        self.n_t_steps = config['n_t_steps']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.lr_factor = config['lr_factor']
        self.lr_patience = config['lr_patience']
        self.n_workers = config['n_workers']
        self.DEVICE = config['device']

        # define PyTorch model
        self.U = torch.nn.Parameter(torch.randn(self.n_units, self.temporal_rank))
        self.V = torch.nn.Parameter(torch.randn(self.temporal_rank, self.n_t_steps, self.n_t_steps))
        self.b = torch.nn.Parameter(torch.randn(self.n_t_steps,))
        self.double()

        self.r2_score = R2Score(num_outputs=self.n_t_steps, multioutput='uniform_average')

    def forward(self, x):
        self.B = torch.einsum('nr,rtd->ntd', self.U, self.V)
        pred = torch.einsum('ntd,ktn->kd', self.B, x)
        pred += self.b
        return pred

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

        # calling self.log will surface up scalars for you in TensorBoard
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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=self.lr_factor, patience=self.lr_patience
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def train_dataloader(self):
        if self.DEVICE.type == 'cuda':
          # setting num_workers > 0 triggers errors so leave it as it is for now
          data_loader = DataLoader(
              self.train_data, batch_size=config['batch_size'], shuffle=True, #num_workers=self.n_workers, pin_memory=True
          )
        else:
          data_loader = DataLoader(self.train_data, batch_size=config['batch_size'], shuffle=True)
        return data_loader

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=config['batch_size'], shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=config['batch_size'], shuffle=False, drop_last=True)

