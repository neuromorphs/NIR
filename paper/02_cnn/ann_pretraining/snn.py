import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
import sinabs.layers as sl
# import sinabs.exodus.layers as sel


class SNN(pl.LightningModule):
    def __init__(self, batch_size, lr=1e-3):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        backend = sl
        self.model = nn.Sequential(
            nn.Conv2d(2, 20, 5, 1, bias=False),
            backend.IAFSqueeze(shape=[batch_size, 20, 30, 30], batch_size=batch_size),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(20, 32, 5, 1, bias=False),
            backend.IAFSqueeze(shape=[batch_size, 32, 11, 11], batch_size=batch_size),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 128, 3, 1, bias=False),
            backend.IAFSqueeze(shape=[batch_size, 128, 3, 3], batch_size=batch_size),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128, 500, bias=False),
            backend.IAFSqueeze(shape=[batch_size, 500], batch_size=batch_size),
            nn.Linear(500, 10, bias=False),
        )

    def forward(self, x):
        self.reset_states()
        return self.model(x.flatten(0, 1)).unflatten(0, (self.batch_size, -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).sum(1)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).sum(1)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)
        prediction = (y_hat.argmax(1) == y).float()
        self.log('valid_acc', prediction.sum() / len(prediction), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).sum(1)
        prediction = (y_hat.argmax(1) == y).float()
        self.log('test_acc', prediction.sum() / len(prediction), prog_bar=True)

    @property
    def sinabs_layers(self):
        return [
            layer
            for layer in self.model.modules()
            if isinstance(layer, sl.StatefulLayer)
        ]

    def reset_states(self):
        for layer in self.sinabs_layers:
            layer.reset_states()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)