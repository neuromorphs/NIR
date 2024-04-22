import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F


class CNN(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Conv2d(2, 20, 5, 1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(20, 32, 5, 1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 128, 3, 1, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128, 500, bias=False),
            nn.ReLU(),
            nn.Linear(500, 10, bias=False),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)
        prediction = (y_hat.argmax(1) == y).float()
        self.log('valid_acc', prediction.sum() / len(prediction), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        prediction = (y_hat.argmax(1) == y).float()
        self.log('test_acc', prediction.sum() / len(prediction), prog_bar=True)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)