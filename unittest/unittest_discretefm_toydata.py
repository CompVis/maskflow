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


def pad_like_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *(1 for _ in range(y.ndim - x.ndim)))


def x2prob(x: Img, vocab_size: int) -> Prob:
    x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
    return rearrange(x, "b h w c -> b c h w")


def sample_p(pt: Prob) -> Img:
    b, _, h, w = pt.shape
    pt = rearrange(pt, "b c h w -> (b h w) c")
    xt = torch.multinomial(pt, 1)
    return xt.reshape(b, h, w)


class Coupling:
    def __init__(self) -> None:
        pass

    def sample(self, x1: Img) -> tuple[Img, Img]:
        raise NotImplementedError


class Ucoupling(Coupling):
    def __init__(self) -> None:
        pass

    def sample(self, x1: Img) -> tuple[Img, Img]:
        return torch.zeros_like(x1), x1


class Ccoupling(Coupling):
    def __init__(self, msk_prop: float = 0.8) -> None:
        self.msk_prob = msk_prop

    def sample(self, x1: Img) -> tuple[Img, Img]:
        I = torch.rand_like(x1.float()) > self.msk_prob
        x0 = x1 * I + torch.zeros_like(x1) * (~I)
        return x0, x1


class KappaScheduler:
    def __init__(self) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        raise NotImplementedError

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        raise NotImplementedError


class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
        self.a = a
        self.b = b

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return (
            -2 * (t**3)
            + 3 * (t**2)
            + self.a * (t**3 - 2 * t**2 + t)
            + self.b * (t**3 - t**2)
        )

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return (
            -6 * (t**2)
            + 6 * t
            + self.a * (3 * t**2 - 4 * t + 1)
            + self.b * (3 * t**2 - 2 * t)
        )


def sample_cond_pt(
    p0: Prob, p1: Prob, t: torch.Tensor | float, kappa: KappaScheduler
) -> Img:
    t = t.reshape(-1, 1, 1, 1)
    pt = (1 - kappa(t)) * p0 + kappa(t) * p1
    return sample_p(pt)


class DiscreteFM:
    def __init__(
        self,
        vocab_size: int,
        model: nn.Module,
        coupling: Coupling,
        kappa: KappaScheduler,
    ) -> None:
        self.vocab_size = vocab_size
        self.model = model
        self.coupling = coupling
        self.kappa = kappa

    def forward_u(self, t: float | torch.Tensor, xt: Img) -> Prob:
        dirac_xt = x2prob(xt, self.vocab_size)
        p1t = torch.softmax(self.model(xt, t.flatten()), dim=1)
        kappa_coeff = self.kappa.derivative(t) / (1 - self.kappa(t))
        return kappa_coeff * (p1t - dirac_xt)

    def backward_u(self, t: float | torch.Tensor, xt: Img) -> Prob:
        dirac_xt = x2prob(xt, self.vocab_size)
        x0 = torch.zeros_like(xt)
        p = x2prob(x0, self.vocab_size)
        kappa_coeff = self.kappa.derivative(t) / self.kappa(t)
        return kappa_coeff * (dirac_xt - p)

    def bar_u(
        self,
        t: float | torch.Tensor,
        xt: Img,
        alpha_t: float | torch.Tensor,
        beta_t: float | torch.Tensor,
    ) -> Prob:
        return alpha_t * self.forward_u(t, xt) - beta_t * self.backward_u(t, xt)

    def training_loss(self, x) -> torch.Tensor:
        if self.input_tensor_type == "bt":
            pass
        elif self.input_tensor_type == "bwh":
            x = rearrange(x, "b c h w -> b h w c")
        else:
            raise ValueError(f"Unknown tensor type: {self.input_tensor_type}")
        x0, x1 = self.coupling.sample(x)
        t = torch.rand(len(x0), device=x.device)
        dirac_x0 = x2prob(x0, self.vocab_size)
        dirac_x1 = x2prob(x1, self.vocab_size)
        xt = sample_cond_pt(dirac_x0, dirac_x1, t, self.kappa)
        p1t = self.model(xt, t)
        loss = F.cross_entropy(p1t, x1.long())
        return loss


class DiscreteSampler:
    def __init__(self, adaptative: bool = True) -> None:
        self.h = self.adaptative_h if adaptative else self.constant_h

    def u(self, t: float | torch.Tensor, xt: Img, discretefm: DiscreteFM) -> Prob:
        raise NotImplementedError

    def adaptative_h(
        self, h: float | torch.Tensor, t: float | torch.Tensor, discretefm: DiscreteFM
    ) -> float | torch.Tensor:
        raise NotImplementedError

    def constant_h(
        self, h: float | torch.Tensor, t: float | torch.Tensor, discretefm: DiscreteFM
    ) -> float | torch.Tensor:
        return h

    def sample_x0(
        self, sample_size: Tuple[int], device: torch.device, vocab_size: int
    ) -> Tuple[Img, Prob]:
        x0 = torch.zeros(sample_size, device=device, dtype=torch.long)
        dirac_x0 = x2prob(x0, vocab_size)
        return x0, dirac_x0

    def __call__(
        self,
        sample_size: Tuple[int],
        discretefm: DiscreteFM,
        n_steps: int,
        t_min: float = 1e-4,
    ) -> List[Img]:
        t = t_min * torch.ones(sample_size[0], device=discretefm.device)
        default_h = 1 / n_steps
        xt, dirac_xt = self.sample_x0(
            sample_size, discretefm.device, discretefm.vocab_size
        )
        list_xt = [xt]
        t = pad_like_x(t, dirac_xt)

        while t.max() <= 1 - default_h:
            h = self.h(default_h, t, discretefm)
            pt = dirac_xt + h * self.u(t, xt, discretefm)
            xt = sample_p(pt)
            dirac_xt = x2prob(xt, discretefm.vocab_size)
            t += h
            list_xt.append(xt)

        return list_xt


class SimpleSampler(DiscreteSampler):
    def u(self, t: float | torch.Tensor, xt: Img, discretefm: DiscreteFM) -> Prob:
        return discretefm.forward_u(t, xt)

    def adaptative_h(
        self, h: float | torch.Tensor, t: float | torch.Tensor, discretefm: DiscreteFM
    ) -> float | torch.Tensor:
        coeff = (1 - discretefm.kappa(t)) / discretefm.kappa.derivative(t)
        h = torch.tensor(h, device=discretefm.device)
        h_adapt = torch.minimum(h, coeff)
        return h_adapt


class CorrectorSampler(DiscreteSampler):
    def __init__(
        self,
        adaptative: bool = True,
        alpha: float = 12.0,
        a: float = 2.0,
        b: float = 0.25,
    ) -> None:
        super().__init__(adaptative)
        self.alpha = alpha
        self.a, self.b = a, b
        self.alpha_t = lambda t: 1 + (self.alpha * (t**self.a)) * ((1 - t) ** self.b)
        self.beta_t = lambda t: self.alpha_t(t) - 1

    def u(self, t: float | torch.Tensor, xt: Img, discretefm: DiscreteFM) -> Prob:
        return discretefm.bar_u(t, xt, self.alpha_t(t), self.beta_t(t))

    def adaptative_h(
        self, h: float | torch.Tensor, t: float | torch.Tensor, discretefm: DiscreteFM
    ) -> float | torch.Tensor:
        alpha_term = (
            self.alpha_t(t) * discretefm.kappa.derivative(t) / (1 - discretefm.kappa(t))
        )
        beta_term = (
            self.beta_t(t) * discretefm.kappa.derivative(t) / discretefm.kappa(t)
        )
        coeff = 1 / (alpha_term + beta_term)
        h = torch.tensor(h, device=discretefm.device)
        h_adapt = torch.minimum(h, coeff)
        return h_adapt


class DiscreteFM_PL(L.LightningModule):
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

    wandb.init(project="discrete-diffusion-mnist-toy", sync_tensorboard=True)

    model = SongUnet(
        img_resolution=img_resolution,
        in_channels=img_channels,
        out_channels=vocab_size,
        model_channels=model_dim,
    )
    coupling, kappa = Ucoupling(), CubicScheduler()
    discretefm = DiscreteFM_PL(vocab_size, model, coupling, kappa)

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
