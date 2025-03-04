# Adapted from DiT and OpenSora

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT:      https://github.com/facebookresearch/DiT
# OpenSora: https://github.com/hpcaitech/Open-Sora
# --------------------------------------------------------


from einops import rearrange
import torch
from torchvision.io import write_video
from torchvision.utils import save_image
from torch import nn


class PatchEmbed_VideoMamba(
    nn.Module
):  # https://github.com/OpenGVLab/VideoMamba/blob/f3427e42cb8453a523aec3a6f86d57b5bc1de5c3/videomamba/video_sm/models/videomamba.py#L174
    """Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x):
        x = rearrange(x, "b t c h w -> b c t h w")
        x = self.proj(x)
        x = rearrange(x, "b c t h w -> b t (h w) c")
        return x


def save_video_cthw(x, save_path, fps=8, normalize=True, value_range=(-1, 1)):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4

    if x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))

        x = (
            x.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 3, 0)
            .to("cpu", torch.uint8)
        )
        write_video(save_path, x, fps=fps, video_codec="h264")
    print(f"Saved to {save_path}")
