# Imports
import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import os
import matplotlib.pyplot as plt
from typing import Tuple, List
from einops import rearrange

# Assuming these are defined in separate files
from discretefm.datamodule import DigitDataModule
from discretefm.unet import SongUnet
from discretefm.utils import create_animation, plot_generation


Prob = torch.Tensor
Img = torch.Tensor


class Main_PL(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        backbone: nn.Module,
        coupling: Coupling,
        kappa: KappaScheduler,
    ) -> None:
        super().__init__()
        self.backward = backbone
        self.dynamic = DiscreteFM(
            vocab_size=vocab_size, model=backbone, coupling=coupling, kappa=kappa
        )

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.dynamic.training_loss(batch[0])

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self.step(batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-3)


# Main execution
if __name__ == "__main__":
    vocab_size = 10
    model_dim = 64
    img_resolution = 32
    img_channels = 1
    batch_size = 200
    n_epochs = 1
    n_steps = 100
    import wandb

    wandb.init(project="discrete-diffusion-mnist-toy-sadd", sync_tensorboard=True)

    model = SongUnet(
        img_resolution=img_resolution,
        in_channels=img_channels,
        out_channels=vocab_size,
        model_channels=model_dim,
    )
    coupling, kappa = Ucoupling(), CubicScheduler()
    discretefm = Main_PL(vocab_size, model, coupling, kappa)

    # Generation / Sampling
    sampler = SimpleSampler()  # or CorrectorSampler()
    sample_size = (16, img_resolution, img_resolution)

    #####
    dm = DigitDataModule(dict_size=vocab_size, batch_size=batch_size)
    dm.setup()
    trainer = L.Trainer(
        max_epochs=n_epochs, logger=TensorBoardLogger("lightning_logs"), devices=1
    )

    trainer.fit(discretefm, dm)

    xts = sampler(sample_size, discretefm, n_steps)
    plot_generation(xts, n_plots=5)
    create_animation(xts, "digits.gif", duration=5, vocab_size=vocab_size)
