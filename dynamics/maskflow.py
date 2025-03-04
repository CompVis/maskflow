# Imports
from typing import Optional

import torch
import math
import numpy as np
import random
from torch import nn
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List
from einops import rearrange
from utils_common import print_rank_0


Prob = torch.Tensor
Img = torch.Tensor


def pad_like_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *(1 for _ in range(y.ndim - x.ndim)))

def get_uniform_n_samples(_input, chain_num):
    chain_indices = (np.linspace(0, 0.9999, chain_num) * len(_input)).astype(np.int32)
    # use 0.9999 instead of 1 for corner case
    _input = [_input[i] for i in chain_indices]
    return _input

def argmax_p(pt, xt, mask_token_id):

    pt[:, mask_token_id, :, :, :] = 0  # make mask_token_id never be the max
    max_xt = pt.argmax(dim=1)
    is_mask = xt == mask_token_id
    xt[is_mask] = max_xt[is_mask]
    _ratio = (is_mask.sum() / is_mask.numel()).item()
    return xt

def adap_scheduler(step, token_num, mode="arccos", leave=False):
    """Create a sampling scheduler
    :param
     step  -> int:  number of prediction during inference
     mode  -> str:  the rate of value to unmask
     leave -> bool: tqdm arg on either to keep the bar or not
    :return
     scheduler -> torch.LongTensor(): the list of tokens to predict at each step
    """
    r = torch.linspace(1, 0, step)
    if mode == "root":  # root scheduler
        val_to_mask = 1 - (r**0.5)
    elif mode == "linear":  # linear scheduler
        val_to_mask = 1 - r
    elif mode == "square":  # square scheduler
        val_to_mask = 1 - (r**2)
    elif mode == "cosine":  # cosine scheduler
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":  # arc cosine scheduler
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
    else:
        return

    sche = (val_to_mask / val_to_mask.sum()) * token_num
    sche = sche.round()
    sche[sche == 0] = 1  
    sche[-1] += (token_num) - sche.sum() 
    return sche.int()


def indices_to_diracp(x: Img, vocab_size: int, data_type: str = "bt") -> Prob:
    assert torch.all(x >= 0) and torch.all(x < vocab_size), f"Indices out of bounds: min {x.min().item()}, max {x.max().item()}, vocab_size {vocab_size}"
    if data_type == "bt":
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b t k -> b k t")
    elif data_type == "bwh":
        b, w, h = x.shape
        x = rearrange(x, "b w h -> b (w h)")
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b (w h) k -> b k w h", w=w, h=h)
    elif data_type == "bcwh":
        b, c, w, h = x.shape
        x = rearrange(x, "b c w h -> b (c w h)")
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b (c w h) k -> b k c w h", c=c, w=w, h=h)
    elif data_type == "btwh":
        b, t, w, h = x.shape
        x = rearrange(x, "b t w h -> b t (w h)")
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b t (w h) k -> b t k w h", w=w, h=h)
    elif data_type == "btcwh":
        b, t, c, w, h = x.shape
        x = rearrange(x, "b t c w h -> b t (c w h)")
        x = torch.nn.functional.one_hot(x, num_classes=vocab_size)
        return rearrange(x, "b t (c w h) k -> b t k c w h", c=c, w=w, h=h)

    else:
        raise ValueError(f"input_tensor_type {data_type} not supported")


def sample_p(pt: Prob, data_type: str) -> Img:
    if data_type == "bt":
        b, k, t = pt.shape
        pt = rearrange(pt, "b k t -> (b t) k")
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, t)
    elif data_type == "bwh":
        b, k, h, w = pt.shape
        pt = rearrange(pt, "b k h w -> (b h w) k")
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, h, w)
    elif data_type == "bcwh":
        b, k, c, h, w = pt.shape
        pt = rearrange(pt, "b  k c h w -> (b c h w) k")
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, c, h, w)
    elif data_type == "btwh":
        b, t, c, h, w = pt.shape
        pt = rearrange(pt, "b t c h w -> (b t h w) c")  
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, t, h, w)
    elif data_type == "btcwh":
        b, t, k, c, w, h = pt.shape
        pt = rearrange(pt, "b t k c w h -> (b t c w h) k")
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, t, c, w, h)
    else:
        raise ValueError(f"input_tensor_type {data_type} not supported")

@torch.compile()
def logits_with_top_k_top_p_(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    data_type: str = None,
) -> torch.Tensor:  # return idx, shaped (B, l)
    if data_type == "bwh":
        b, k, w, h = logits_BlV.shape
        logits_BlV = rearrange(logits_BlV, "b k w h -> b (w h) k")
    elif data_type in ["bcwh", "btwh"]:
        b, k, c, w, h = logits_BlV.shape
        logits_BlV = rearrange(logits_BlV, "b k c w h -> b (c w h) k")
    elif data_type == "btcwh":
        b, k, t, c, w, h = logits_BlV.shape
        logits_BlV = rearrange(logits_BlV, "b k t c w h -> b (t c w h) k")
    elif data_type == "bt":
        b, k, t = logits_BlV.shape
        logits_BlV = rearrange(logits_BlV, "b k t -> b t k")
    else:
        raise ValueError(f"data_type={data_type} not supported")
    if top_k > 0:
        idx_to_remove = logits_BlV < logits_BlV.topk(
            top_k, largest=True, sorted=False, dim=-1
        )[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (
            1 - top_p
        )
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(
            sorted_idx_to_remove.scatter(
                sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove
            ),
            -torch.inf,
        )
    if data_type == "bwh":
        logits_BlV = rearrange(logits_BlV, "b (w h) k -> b k w h", w=w, h=h)
    elif data_type in ["bcwh", "btwh"]:
        logits_BlV = rearrange(logits_BlV, "b (c w h) k -> b k c w h", c=c, w=w, h=h)
    elif data_type == "bt":
        logits_BlV = rearrange(logits_BlV, "b t k -> b k t")
    elif data_type == "btcwh":
        logits_BlV = rearrange(logits_BlV, "b (c t w h) k -> b k t c w h", c=c, t=t, w=w, h=h)
    else:
        raise ValueError(f"data_type={data_type} not supported")
    return logits_BlV


class KappaScheduler:
    def __init__(self) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        raise NotImplementedError

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        raise NotImplementedError


class Coupling:
    def __init__(self) -> None:
        pass

    def sample(self, x1: Img) -> tuple[Img, Img]:
        raise NotImplementedError


class Ucoupling(Coupling):
    def __init__(self, mask_token_id) -> None:
        self.mask_token_id = mask_token_id

    def sample(self, x1: Img) -> tuple[Img, Img]:
        return torch.ones_like(x1) * self.mask_token_id, x1


class Ccoupling(Coupling):
    def __init__(self, mask_token_id: int, msk_prop: float = 0.8) -> None:
        if msk_prop is None:
            print("Ccoupling, msk_prop is None, using coupling by random prob")
        elif msk_prop > 0:
            print("Ccoupling, msk_prop: ", msk_prop, "data_prob", 1 - msk_prop)
        else:
            raise ValueError("msk_prop must be non-negative")
        self.mask_token_id = mask_token_id
        self.msk_prob = msk_prop

    def sample(self, x1: Img) -> tuple[Img, Img]:
        if self.msk_prob is None:
            _msk_prob = torch.rand_like(x1.float())
        else:
            _msk_prob = self.msk_prob
        _mask20 = torch.rand_like(x1.float()) > _msk_prob
        _mask_id = torch.ones_like(x1) * self.mask_token_id
        x0 = x1 * _mask20 + _mask_id * (~_mask20)
        return x0, x1


class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 0.0, b: float = 2.0) -> None:
        self.a = a
        self.b = b

    def __call__(
        self, t: float | torch.Tensor
    ) -> float | torch.Tensor:  # Eq 33 in paper
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


class LinearScheduler(KappaScheduler):
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1.0


class QuadraticScheduler(KappaScheduler):
    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t**2

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 2 * t


class RootScheduler(KappaScheduler):
    def __init__(self) -> None:
        pass

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return t**0.5

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 0.5 * t ** (-0.5)


class CosineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = torch.pi * 0.5

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1 - torch.cos(self.coeff * t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.sin(self.coeff * t) * self.coeff


class SineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = torch.pi * 0.5

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.sin(self.coeff * t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return torch.cos(self.coeff * t) * self.coeff


class ArcCosineScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = 2 / torch.pi
        self.eps = 1e-6

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return 1 - self.coeff * torch.acos(t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return self.coeff / torch.sqrt(1 - t**2 + self.eps)


class ArcSinScheduler(KappaScheduler):
    def __init__(self) -> None:
        self.coeff = 2 / torch.pi
        self.eps = 1e-6

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return self.coeff * torch.asin(t)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return self.coeff / torch.sqrt(1 - t**2 + self.eps)

class SigmoidScheduler(KappaScheduler):
    def __init__(self, start=-3.0, end=3.0, tau=1.0, clip_min=1e-5):
        self.start = torch.tensor(start)
        self.end = torch.tensor(end)
        self.tau = torch.tensor(tau)
        self.clip_min = clip_min
        self.v_start = torch.sigmoid(self.start / self.tau)
        self.v_end = torch.sigmoid(self.end / self.tau)
    
    def __call__(self, t):
        output = torch.sigmoid((t * (self.end - self.start) + self.start) / self.tau)
        output = (self.v_end - output) / (self.v_end - self.v_start)
        return torch.clamp(1 - output, min=self.clip_min, max=1.0)
    
    def derivative(self, t):
        sigmoid_val = torch.sigmoid((t * (self.end - self.start) + self.start) / self.tau)
        return (self.end - self.start) / self.tau * sigmoid_val * (1 - sigmoid_val) / (self.v_end - self.v_start)

class DiscreteFM:
    def __init__(
        self,
        vocab_size: int,
        coupling: Coupling,
        kappa: KappaScheduler,
        device: torch.device,
        objective: str = "pred_x0",
        noise_level: str = "random_all",
        input_tensor_type: str = "btwh",
        reweigh_loss: str = "snr",
        sampling_timesteps: int = 250,
        smoothing_factor: float = 0.0,
        uncertainty_scale: float = 1.0,
        temperature: float = 1.0,
        snr_clip: float = 6.0,
        cum_snr_decay: float = 0.96,
        use_fp16: bool = False,
        use_fused_snr: bool = True,
        mask_ce=False,
    ) -> None:
        self.vocab_size = vocab_size
        self.coupling = coupling
        self.kappa = kappa
        self.device = device
        self.data_type = input_tensor_type
        self.smoothing_factor = smoothing_factor
        self.mask_ce = mask_ce
        self.sampling_timesteps = sampling_timesteps
        self.temperature = temperature
        self.use_fp16 = use_fp16
        self.cum_snr_decay = cum_snr_decay
        self.snr_clip = snr_clip
        self.reweigh_loss = reweigh_loss
        self.use_fused_snr = use_fused_snr
        self.objective = objective
        self.eps = 1e-6

    def forward_u(
            self, t: float | torch.Tensor, xt: Img, model: nn.Module, **model_kwargs
        ) -> Prob:  # Eq 24 and Eq 57

        dirac_x0, _ = self.coupling.sample(xt)
        dirac_xt = indices_to_diracp(
            xt,
            self.vocab_size,
            self.data_type
        )
        if self.data_type == "btwh":
            batch_size, n_frames, h, w = xt.shape
            dirac_xt = rearrange(dirac_xt, "b t k w h -> b k t w h")
        elif self.data_type == "btcwh":
            batch_size, n_frames, n_channels, h, w = xt.shape
            dirac_xt = rearrange(dirac_xt, "b t k c w h -> b k t c w h")
        else:
            raise ValueError(f"data type {self.data_type} not supported.")

        logits = model(xt, t, **model_kwargs) / self.temperature
        p1t = torch.softmax(logits, dim=1)
        kappa_coeff = self.kappa.derivative(t) / (1 - self.kappa(t) + self.eps) 
        kappa_coeff = kappa_coeff.reshape(batch_size, 1, n_frames, 1, 1)
        
        if self.data_type == "btcwh":
            kappa_coeff = kappa_coeff.unsqueeze(-1)

        u = kappa_coeff * (p1t - dirac_xt)
        return u

    def forward_u_mgm(
        self,
        t: float | torch.Tensor,
        xt: Img,
        model: nn.Module,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        data_type: str = None,
        **model_kwargs,
    ) -> Prob:  # Eq 24 and Eq 57

        logits = model(xt, t, **model_kwargs)
        assert torch.isfinite(logits).all()

    
        if temperature is not None and temperature > 0:
            logits = logits / temperature

        logits = logits_with_top_k_top_p_(
            logits, top_k=top_k, top_p=top_p, data_type=self.data_type
        )

        if self.data_type == "btwh":
            b, k, t, w, h = logits.shape
            _logits = rearrange(logits, "b k t w h -> b (t w h) k")
        else:
            _logits = logits
        
        assert torch.isfinite(_logits).all()

        p1t = torch.softmax(logits, dim=1)
        return p1t, _logits

    def corrupt_data(
        self,
        p0: Prob,
        p1: Prob,
        t: torch.Tensor | float,
        kappa: KappaScheduler,
        data_type: str,
    ) -> Img:
        p0_shape = p0.shape
        pt = (1 - kappa(t)) * p0 + kappa(t) * p1

        assert torch.all(pt >= 0), "Negative probabilities in pt"

        if self.data_type == "btwh":
            assert torch.allclose(pt.sum(dim=2), torch.ones_like(pt.sum(dim=2)), atol=1e-6), "Probabilities do not sum to 1"

        if self.smoothing_factor > 0.0:
            pt = pt + self.smoothing_factor * (1 - kappa(t)) * kappa(t)

        return sample_p(pt, data_type)

    def corrupt_data_indices(self, x, t):

        if self.data_type == "btwh":
            batch_size, n_frames, _, _ = x.shape
            t = t.view(batch_size, n_frames, 1, 1, 1)
        elif self.data_type == "btcwh":
            batch_size, n_frames, n_channels, _, _ = x.shape
            t = t.view(batch_size, n_frames, 1, 1, 1, 1)

        x0, x1_target = self.coupling.sample(x)

        dirac_x0 = indices_to_diracp(x0.long(), self.vocab_size, self.data_type)  
        dirac_x1 = indices_to_diracp(x1_target.long(), self.vocab_size, self.data_type)
        
        return self.corrupt_data(dirac_x0, dirac_x1, t, self.kappa, self.data_type)


    def training_losses(self, model, x, noise_levels) -> torch.Tensor:
        """
        Compute the training loss using individual noise levels for each frame.

        Args:
            model (nn.Module): The model to train.
            x (torch.Tensor): Input tensor of shape (b, t, h, w) or (b, t, c, h, w) if data_type == "btcwh".
            noise_levels (torch.Tensor): Noise levels tensor of shape (b, t).

        Returns:
            dict: A dictionary containing loss, logits, corrupted inputs, and metrics.
        """

        if self.data_type == "btwh":
            bs, n_frames, w, h = x.shape
        elif self.data_type == "btcwh":
            bs, n_frames, c, w, h = x.shape

        device = x.device
        x0, x1_target = self.coupling.sample(x)
        dirac_x0 = indices_to_diracp(x0.long(), self.vocab_size, self.data_type)
        dirac_x1 = indices_to_diracp(x1_target.long(), self.vocab_size, self.data_type)

        t = noise_levels.view(bs, n_frames, 1, 1, 1)
        if self.data_type == "btcwh":
            t = t.unsqueeze(-1)

        xt = self.corrupt_data(dirac_x0, dirac_x1, t, self.kappa, self.data_type)

        logits_x = model(
            x=xt,
            t=noise_levels,
            use_fp16=self.use_fp16,
            cond_drop_prob=0.0,
        )

        if self.objective == "pred_x0":
            loss = F.cross_entropy(
                logits_x,
                x1_target.long(),
                reduction="none"
            )
        else:
            raise ValueError(f"unknown objective {self.objective}")

        

        if self.reweigh_loss == "snr":
            loss_weights = self.compute_loss_weights(noise_levels)
            loss_weights = loss_weights.reshape(*loss_weights.shape, 1, 1)
            if self.data_type == "btcwh":
                loss_weights = loss_weights.unsqueeze(-1)

            loss = loss * loss_weights

        target_mask = (xt != x1_target).float()
        target_mask_flat = target_mask.reshape(-1)
        loss_flat = loss.reshape(-1)
        mask_sum = target_mask_flat.sum().item()

        if self.mask_ce:
            loss = (loss_flat * target_mask_flat).sum() / (mask_sum + self.eps)
        else:
            loss = loss.mean()

        preds = logits_x.argmax(dim=1)
        preds_flat = preds.view(-1)
        x1_target_flat = x1_target.view(-1)
        acc = ((preds_flat == x1_target_flat).float() * target_mask_flat).sum() / (mask_sum + self.eps)
        

        ret_dict = {
            "loss": loss,
            "logits": logits_x,
            "x_corrupt": xt,
            "log/mask_ce": int(self.mask_ce),
            "log/acc": acc,
        }

        return ret_dict

    
    def compute_loss_weights(self, noise_levels: torch.Tensor):
        snr = (self.kappa(noise_levels)**2) / ((1 - self.kappa(noise_levels))**2)
        clipped_snr = torch.clamp(snr, max=self.snr_clip)
        normalized_clipped_snr = clipped_snr / self.snr_clip
        normalized_snr = snr / self.snr_clip

        if not self.use_fused_snr:
            match self.objective:
                case "pred_x0":
                    return clipped_snr
                case _:
                    raise ValueError(f"unknown objective {self.objective}")

        cum_snr = torch.zeros_like(normalized_snr)
        for t in range(0, noise_levels.shape[1]):
            if t == 0:
                cum_snr[:, t] = normalized_clipped_snr[:, t]
            else:
                cum_snr[:, t] = self.cum_snr_decay * cum_snr[:, t - 1] + (1 - self.cum_snr_decay) * normalized_clipped_snr[:, t]

        cum_snr_shifted = F.pad(cum_snr[:, :-1], (1, 0), value=0.0)
        clipped_fused_snr = 1 - (1 - cum_snr_shifted * self.cum_snr_decay) * (1 - normalized_clipped_snr)
        fused_snr = 1 - (1 - cum_snr_shifted * self.cum_snr_decay) * (1 - normalized_snr)

        match self.objective:
            case "pred_x0":
                return clipped_fused_snr * self.snr_clip
            case _:
                raise ValueError(f"unknown objective {self.objective}")

    def _generate_scheduling_matrix(self, type: str, horizon: int):
        match type:
            case "pyramid":
                return self._generate_pyramid_scheduling_matrix(horizon)
            case "full_sequence":
                return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)

    def _generate_pyramid_scheduling_matrix(self, horizon: int):
        height = self.sampling_timesteps + int((horizon - 1)) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps - 1)


class DiscreteSampler:
    def __init__(self, adaptative: bool = True) -> None:
        self.h = self.adaptative_h if adaptative else self.constant_h

    def u(
        self, t: float | torch.Tensor, xt: Img, discretefm: DiscreteFM, model: nn.Module
    ) -> Prob:
        raise NotImplementedError

    def adaptative_h(
        self, h: float | torch.Tensor, t: float | torch.Tensor, discretefm: DiscreteFM
    ) -> float | torch.Tensor:
        raise NotImplementedError

    def constant_h(
        self, h: float | torch.Tensor, t: float | torch.Tensor, discretefm: DiscreteFM
    ) -> float | torch.Tensor:
        return h

    def construct_x0(
        self, sample_size: Tuple[int], device: torch.device, vocab_size: int
    ) -> Tuple[Img, Prob]:
        x0 = (
            torch.ones(sample_size, device=device, dtype=torch.long)
            * self.mask_token_id
        )
        dirac_x0 = indices_to_diracp(x0, vocab_size, self.data_type)
        return x0, dirac_x0

    def __call__(
        self,
        sample_size: Tuple[int],
        discretefm: DiscreteFM,
        model: nn.Module,
        n_steps: int,
        t_min: float = 1e-4,
        **model_kwargs,
    ) -> List[Img]:
        t = t_min * torch.ones(sample_size[0], device=discretefm.device)
        default_h = 1 / n_steps
        xt, dirac_xt = self.construct_x0(
            sample_size, discretefm.device, discretefm.vocab_size
        )
        list_xt = [xt]
        t = pad_like_x(t, dirac_xt)

        while t.max() <= 1 - default_h:
            h = self.h(default_h, t, discretefm)
            pt = dirac_xt + h * self.u(t, xt, discretefm, model, **model_kwargs)
            xt = sample_p(pt, discretefm.data_type)
            # Eq 12 in https://arxiv.org/pdf/2407.15595
            dirac_xt = indices_to_diracp(
                xt, discretefm.vocab_size, discretefm.data_type
            )
            t += h
            list_xt.append(xt)
        return list_xt


class FMSampler(DiscreteSampler):
    def __init__(
        self, mask_token_id: int, input_tensor_type: str = "bt", adaptive_stepsize=True
    ):
        super().__init__(adaptive_stepsize)
        self.mask_token_id = mask_token_id
        self.data_type = input_tensor_type

    def u(
        self,
        t: float | torch.Tensor,
        xt: Img,
        discretefm: DiscreteFM,
        model: nn.Module,
        **model_kwargs,
    ) -> Prob:
        return discretefm.forward_u(t, xt, model, **model_kwargs)

    def adaptative_h(
        self, h: float | torch.Tensor, t: float | torch.Tensor, discretefm: DiscreteFM
    ) -> float | torch.Tensor:  # Eq 30 in https://arxiv.org/pdf/2407.15595
        coeff = (1 - discretefm.kappa(t)) / discretefm.kappa.derivative(t)
        h = torch.tensor(h, device=discretefm.device)
        h_adapt = torch.minimum(h, coeff)
        return h_adapt

    @torch.no_grad()
    def sample_step_with_noise_schedule(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        discretefm: DiscreteFM,
        model: nn.Module,
        n_steps: int,
        model_kwargs: dict = {}
    ):
        bs, n_frames = xt.shape[:2]

        default_h = 1 / n_steps
        h = self.h(default_h, t, discretefm)
        t = torch.clamp(t, min=default_h, max=1 - default_h)

        _u = self.u(
                t,
                xt,
                discretefm,
                model,
                **model_kwargs,
            ) 

        dirac_xt = indices_to_diracp(
            xt,
            discretefm.vocab_size,
            discretefm.data_type
        )

        if discretefm.data_type == "btwh":
            dirac_xt = rearrange(dirac_xt, "b t k w h -> b k t w h")
            pt = dirac_xt + h * _u # b k t w h
            pt = rearrange(pt, 'b k t w h -> b t k w h')      
        elif discretefm.data_type == "btcwh":
            dirac_xt = rearrange(dirac_xt, "b t k c w h -> b k t c w h")
            pt = dirac_xt + h * _u # b k t c h w
            pt = rearrange(pt, 'b k t c w h -> b t k c w h')
    
        pt = torch.clamp(pt, min=0.0, max=1.0)
        _xt = sample_p(pt, discretefm.data_type) 

        will_unmask = xt == self.mask_token_id 
        xt = torch.where(will_unmask, _xt, xt) 
        return xt

    @torch.no_grad()
    def sample_step_with_partial_context_guidance(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        discretefm: DiscreteFM,
        model: nn.Module,
        n_steps: int,
        n_context: int,
        partial_context_guidance_level: float,
        model_kwargs: dict = {},
    ) -> torch.Tensor:

        batch_size = xt.shape[0]
        default_h = 1 / n_steps
        h = self.h(default_h, t, discretefm)
        t = torch.clamp(t, min=default_h, max=1 - default_h)
        u_cond = self.u(t, xt, discretefm, model, **model_kwargs)

        xt_uncond = xt.clone()
        xt_uncond[:, :n_context] = discretefm.coupling.mask_token_id
        u_uncond = self.u(t, xt_uncond, discretefm, model, **model_kwargs)
        partial_t = torch.full((batch_size, n_context), 0.9, device=xt.device, dtype=t.dtype)
        partial_t = partial_t.view(batch_size, n_context, *(1 for _ in range(xt.ndim - 2)))
        xt_partial = xt.clone()
        xt_partial[:, :n_context] = discretefm.corrupt_data_indices(xt[:, :n_context], partial_t)
        u_partial = self.u(t, xt_partial, discretefm, model, **model_kwargs)
        u_eff = u_cond + partial_context_guidance_level * (u_partial - u_uncond)

        dirac_xt = indices_to_diracp(xt, discretefm.vocab_size, discretefm.data_type)
        if discretefm.data_type == "btwh":
            dirac_xt = rearrange(dirac_xt, "b t k w h -> b k t w h")
            pt = dirac_xt + h * u_eff
            pt = rearrange(pt, "b k t w h -> b t k w h")
        elif discretefm.data_type == "btcwh":
            dirac_xt = rearrange(dirac_xt, "b t k c w h -> b k t c w h")
            pt = dirac_xt + h * u_eff
            pt = rearrange(pt, "b k t c w h -> b t k c w h")
        else:
            raise ValueError(f"Data type {discretefm.data_type} not supported.")

        pt = torch.clamp(pt, min=0.0, max=1.0)
        _xt = sample_p(pt, discretefm.data_type)
        will_unmask = (xt == discretefm.coupling.mask_token_id)
        xt_updated = torch.where(will_unmask, _xt, xt)
        return xt_updated

class MGMSampler:
    def __init__(self, mask_token_id: int, input_tensor_type: str = "bt") -> None:
        self.mask_token_id = mask_token_id
        self.data_type = input_tensor_type

    def logits_squeeze(self, x, data_type: str):
        if data_type == "bt":
            b, k, t = x.shape
            return x.view(b, t, k)
        elif data_type == "bcwh":
            b, k, c, h, w = x.shape
            return rearrange(x, "b k c h w -> b (c w h) k")
        elif data_type == "bwh":
            b, k, h, w = x.shape
            return rearrange(x, "b k h w -> b (h w) k")
        elif data_type == "btwh":
            b, k, t, h, w = x.shape
            return rearrange(x, "b k t h w -> b (t h w) k")
        elif data_type == "btcwh":
            b, k, t, c, w, h = x.shape
            return rearrange(x, "b k t c w h -> b (t c w h) k")
        else:
            raise ValueError(f"data_type={data_type} not supported")

    def sample(
        self,
        sample_size,
        discretefm: DiscreteFM,
        model: nn.Module,
        noise_levels: torch.Tensor,
        n_steps: int,
        init_code=None,
        r_temp=4.5,
        chain_num: int = 20,
        **model_kwargs,
    ):

        temperature = model_kwargs.pop("temperature", 1.0)
        top_p = model_kwargs.pop("top_p", 1.0)
        top_k = model_kwargs.pop("top_k", 0)
        return_chains = model_kwargs.pop("return_chains", 1)
        max_last = model_kwargs.pop("max_last", False)
        mgm_mode = model_kwargs.pop("mgm_mode", "arccos")
        randomize = model_kwargs.pop("mgm_randomize", "none")

        if self.data_type == "btwh":
            batch_size, n_frames, h, w = sample_size
            token_num = n_frames * h * w
        elif self.data_type in ["btcwh", "btchw"]:
            batch_size, n_frames, c, w, h = sample_size
            token_num = n_frames * c * w * h
        else:
            raise ValueError(f"Unsupported data_type {self.data_type}")

        device = discretefm.device

        if init_code is not None:
            code = init_code.long().to(device)
        else:
            code = torch.full(sample_size, self.mask_token_id, device=device, dtype=torch.long)

        code_for_forward = code.clone()
        mask = (code == self.mask_token_id).long()
        mask_flat = mask.view(batch_size, token_num, 1)
        scheduler = adap_scheduler(n_steps, mode=mgm_mode, token_num=token_num)
        code_list = []

        for idx, t in enumerate(scheduler):
            current_mask_sum = mask_flat.sum()
            if current_mask_sum < t:
                t = int(current_mask_sum.item())
            if current_mask_sum == 0:
                break
            _prob, _ = discretefm.forward_u_mgm(
                t=noise_levels,
                xt=code_for_forward,
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                data_type=discretefm.data_type,
                **model_kwargs,
            )
            prob = self.logits_squeeze(_prob, discretefm.data_type)
            distri = torch.distributions.Categorical(probs=prob)
            pred_code = distri.sample()
    
            conf = torch.gather(prob, 2, pred_code.unsqueeze(-1))
            if randomize == "linear":
                ratio = idx / (len(scheduler) - 1)
                rand = r_temp * np.random.gumbel(size=(batch_size, token_num)) * (1 - ratio)
                conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(device)
                conf = rearrange(conf, "b t -> b t 1")
            elif randomize == "warm_up":
                conf = torch.rand_like(conf) if idx < 2 else conf
            elif randomize == "random":
                conf = torch.rand_like(conf)
            elif randomize == "none":
                pass
            else:
                raise ValueError(f"randomize={randomize} not supported")
            conf[~mask_flat.bool()] = -math.inf
            thresh_conf, indice_mask = torch.topk(conf.view(batch_size, -1), k=t, dim=-1)
            thresh_conf = thresh_conf[:, -1]
            conf_mask = (conf >= thresh_conf.unsqueeze(-1).unsqueeze(-1))
            conf_mask = conf_mask.view(batch_size, token_num, 1)
            code_flat = code.view(batch_size, token_num, 1)
            pred_code_flat = pred_code.view(batch_size, token_num, 1)
            f_mask = (mask_flat * conf_mask.float()).bool()
            code_flat[f_mask] = pred_code_flat[f_mask]
            
            if self.data_type == "btwh":
                code = code_flat.view(batch_size, n_frames, h, w)
            else:
                code = code_flat.view(batch_size, n_frames, c, h, w)
            for i_mask, ind_mask in enumerate(indice_mask):
                mask_flat[i_mask, ind_mask, 0] = 0
            code_list.append(code.clone())

        code_list = get_uniform_n_samples(code_list, chain_num)
        code_list = torch.stack(code_list, dim=0).to(device)
        mask_list = torch.zeros_like(code_list, dtype=torch.uint8)
        mask_list[code_list == self.mask_token_id] = 1
        return code_list, mask_list



    @torch.no_grad()
    def sample_with_partial_context_guidance(
        self,
        sample_size,
        discretefm: DiscreteFM,
        model: nn.Module,
        noise_levels: torch.Tensor,
        n_steps: int,
        n_context: int,
        partial_context_guidance_level: float,
        partial_context_guidance_steps:  int,
        init_code=None,
        r_temp=4.5,
        chain_num: int = 7,
        **model_kwargs,
    ):
        temperature = model_kwargs.pop("temperature", 1.0)
        top_p = model_kwargs.pop("top_p", 0.0)
        top_k = model_kwargs.pop("top_k", 0)
        return_chains = model_kwargs.pop("return_chains", 1)
        max_last = model_kwargs.pop("max_last", False)
        mgm_mode = model_kwargs.pop("mgm_mode", "arccos")
        randomize = model_kwargs.pop("mgm_randomize", "none")

        if self.data_type == "btwh":
            batch_size, n_frames, h, w = sample_size
            token_num = n_frames * h * w
            context_token_count = n_context * h * w
        elif self.data_type in ["btcwh", "btchw"]:
            batch_size, n_frames, c, w, h = sample_size
            token_num = n_frames * c * w * h
            context_token_count = n_context * c * w * h
        else:
            raise ValueError(f"Unsupported data_type {self.data_type}")

        device = discretefm.device

        l_codes = []
        l_mask = []

        code = (
            init_code.long().to(device)
            if init_code is not None
            else torch.full(sample_size, self.mask_token_id, device=device)
        )
        
        mask = (code == self.mask_token_id).long()
        mask_flat = mask.view(batch_size, token_num, 1)

        if context_token_count > 0:
            mask_flat[:, :context_token_count, :] = 0

        scheduler = adap_scheduler(n_steps, mode=mgm_mode, token_num=token_num)
        code_list = []

        for indice, t in enumerate(scheduler):
            current_mask_sum = mask_flat.sum()
            if current_mask_sum < t:
                t = int(current_mask_sum.item())
            if current_mask_sum == 0:
                break

            _prob, logits = discretefm.forward_u_mgm(
                t=noise_levels,
                xt=code,
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                data_type=discretefm.data_type,
                **model_kwargs,
            )

            if partial_context_guidance and indice < partial_context_guidance_steps:
                partial_t = torch.full((batch_size, n_context), 0.3, device=device, dtype=noise_levels.dtype)
                partial_t = partial_t.view(batch_size, n_context, *([1] * (code.ndim - 2)))
                code_partial = code.clone()
                code_partial[:, :n_context] = discretefm.corrupt_data_indices(
                    code[:, :n_context], partial_t
                )
                _prob_partial, logits_partial = discretefm.forward_u_mgm(
                    t=noise_levels,
                    xt=code_partial,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    data_type=discretefm.data_type,
                    **model_kwargs,
                )

                code_uncond = code.clone()
                code_uncond[:, :n_context] = discretefm.coupling.mask_token_id
                _prob_uncond, logits_uncond = discretefm.forward_u_mgm(
                    t=noise_levels,
                    xt=code_uncond,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    data_type=discretefm.data_type,
                    **model_kwargs,
                )

                combined_logits = logits + partial_context_guidance_level * (logits_partial - logits_uncond)
            else:
                combined_logits = logits

            combined_logits = combined_logits.clamp(min=-60, max=60) 
            prob = torch.softmax(combined_logits, dim=2)
            distri = torch.distributions.Categorical(probs=prob)
            pred_code = distri.sample()

            conf = torch.gather(prob, 2, pred_code.view(batch_size, token_num, 1))

            if context_token_count > 0:
                conf[:, :context_token_count, :] = -float("inf")

            if model_kwargs.get("mgm_randomize", "none") == "linear":
                ratio = indice / (len(scheduler) - 1)
                rand = r_temp * np.random.gumbel(size=(batch_size, token_num)) * (1 - ratio)
                conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(device)
                conf = rearrange(conf, "b t -> b t 1")
            elif model_kwargs.get("mgm_randomize", "none") == "warm_up":
                conf = torch.rand_like(conf) if indice < 2 else conf
            elif model_kwargs.get("mgm_randomize", "none") == "random":
                conf = torch.rand_like(conf)
            elif model_kwargs.get("mgm_randomize", "none") == "none":
                pass
            else:
                raise ValueError(f"mgm_randomize not supported")

            conf[~mask_flat.bool()] = -float("inf")

            conf_reshaped = conf.view(batch_size, -1)  # shape [B, token_num]
            tresh_conf, indice_mask = torch.topk(conf_reshaped, k=t, dim=-1)
            tresh_val = tresh_conf[:, -1].unsqueeze(-1)  # shape [B, 1]

            conf_binary = (conf_reshaped >= tresh_val).view(batch_size, token_num, 1).float()
            f_mask = (mask_flat * conf_binary).bool()  

            code_flat = code.view(batch_size, token_num, 1)
            pred_code_flat = pred_code.view(batch_size, token_num, 1)
            code_flat[f_mask] = pred_code_flat[f_mask]

            if self.data_type == "btwh":
                code = code_flat.view(batch_size, n_frames, h, w)
            else:
                code = code_flat.view(batch_size, n_frames, c, h, w)

            for b_i, mask_indices in enumerate(indice_mask):
                mask_flat[b_i, mask_indices, 0] = 0

            l_codes.append(pred_code_flat.clone())
            l_mask.append(mask_flat.clone())
            code_list.append(code.clone())

        code_list = get_uniform_n_samples(code_list, chain_num)
        code_list = torch.stack(code_list, dim=0).to(device)
        mask_list = torch.zeros_like(code_list, dtype=torch.uint8)
        mask_list[code_list == self.mask_token_id] = 1
        return code_list, mask_list


    

if __name__ == "__main__":
    pass
