import torch
import tonic
import norse
import nir
import lightning.pytorch as pl

# %%
g = nir.read("cnn_sinabs.nir")

# %%
model = norse.torch.from_nir(g)

# %%
to_frame = tonic.transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size,
        time_window = 1e3)
dataset = tonic.datasets.NMNIST(".", transform=to_frame)

# %%
loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=10, 
        collate_fn=tonic.collation.PadTensors(batch_first=False))

# %%
events, label = next(iter(loader))
events.shape, label.shape

# %%
children = list(model.children())

# %%
a, b = model(events[0])
a.shape

# %%

class CNNModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def training_step(self, batch, batch_idx):
        xs, label = batch
        state = None
        for x in xs:
            out, state = self.model(x, state)
        agg = out.mean(0)
        loss = torch.nn.functional.cross_entropy(agg, label.float())
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
cnn = CNNModel(model)
logger = pl.loggers.TensorBoardLogger(".")
trainer = pl.Trainer(max_epochs=100, accelerator="gpu", logger=logger)
trainer.fit(model=cnn, train_dataloaders=loader)

# %%
