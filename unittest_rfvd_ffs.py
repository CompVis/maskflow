import sys

sys.path.append("..")

import os
import argparse
import numpy as np
import shutil

from tqdm import tqdm
import torch
import os
import torch
import torchvision
import io
from einops import rearrange
from torchvision import transforms
import random
import PIL
from omegaconf import OmegaConf
from fvd_external import calculate_fvd_github

try:
    from datasets_wds.video_utils import (
        TemporalRandomCrop,
        RandomHorizontalFlipVideo,
        ToTensorVideo,
        UCFCenterCropVideo,
    )
except ImportError:
    from video_utils import (
        TemporalRandomCrop,
        RandomHorizontalFlipVideo,
        ToTensorVideo,
        UCFCenterCropVideo,
    )
from utils.my_metrics_offline_video import calc_metrics_for_dataset

source_train = "~/data/preprocess_ffs/train/videos"
source_train = os.path.expanduser(source_train)
fvd_video_dir = "./data/unittest_ffs_tokenizer"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

resolution = 256
num_frames = 16
frame_interval = 4
fvd_frames = 16
fvd_video_num = 1024
write_fps = 8  # https://github.com/FoundationVision/OmniTokenizer/blob/701b619003b3e941e769269c7626dbf111d0377e/Diffusion/Latte/datasets/sky_image_datasets.py#L136
is_debug = False
latent_size = 32

temporal_sample = TemporalRandomCrop(fvd_frames * frame_interval)

tokenizer_type = "sd_vq"
assert tokenizer_type in ["sd_vq", "titok"]

fvd_video_dir = fvd_video_dir + "_" + tokenizer_type

from titok_1d_tokenizer.modeling.titok import TiTok


def titok_tokenizer(tokenizer_name="titok_s128"):
    if tokenizer_name == "titok_s128":
        _tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_s128_imagenet")
    elif tokenizer_name == "titok_l32":
        _tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet")
    else:
        raise ValueError(f"tokenizer={tokenizer_name} not supported")
    _tokenizer.eval()
    _tokenizer.requires_grad_(False)
    _tokenizer = _tokenizer.to(device)

    @torch.no_grad()
    def tokenizer_encode_fn(img):
        img = img / 255.0
        x = _tokenizer.encode(img)[1]["min_encoding_indices"]
        x = x.squeeze(1)
        return x

    @torch.no_grad()
    def tokenizer_decode_fn(indices):
        indices = indices.unsqueeze(1)
        img = _tokenizer.decode_tokens(indices)
        # use the sample batch size, as this is transformer based backbone, it's not so memory-consuming as UNet
        img = torch.clamp(img, 0.0, 1.0)
        img = (img * 255.0).to(dtype=torch.uint8)
        return img

    return tokenizer_encode_fn, tokenizer_decode_fn


def SD_VQ_tokenizer():
    sys.path.insert(0, os.path.abspath("./ldm"))
    from ldm.ldm.util import instantiate_from_config

    ckpt_path = "./pretrained_ckpt/ldm/vq-f8.ckpt"
    config_path = "./ldm/models/first_stage_models/vq-f8/config.yaml"

    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"]
    _tokenizer = instantiate_from_config(config.model)
    _tokenizer.load_state_dict(sd, strict=False)
    _tokenizer.eval()
    _tokenizer.requires_grad_(False)
    _tokenizer = _tokenizer.to(device)

    @torch.no_grad()
    def tokenizer_encode_fn(img, mini_bs=25):
        img = img / 255.0
        img = (img - 0.5) * 2
        # somelogic about video
        img_shape = img.shape
        if len(img_shape) == 5:
            b, t, c, h, w = img.shape
            img = rearrange(img, "b t c h w -> (b t) c h w")
        ############################################################
        for i in range(0, len(img), mini_bs):
            _img = img[i : i + mini_bs]
            encode_res = _tokenizer.encode(_img)
            quant = encode_res[0]
            diff = encode_res[1]
            _indices = encode_res[2][-1]
            if i == 0:
                indices = _indices
            else:
                indices = torch.cat([indices, _indices], dim=0)
        ############################################################
        if len(img_shape) == 5:
            indices = rearrange(
                indices,
                "(b t h w) -> b t h w",
                b=b,
                t=t,
                h=latent_size,
                w=latent_size,
            )
        elif len(img_shape) == 4:
            indices = rearrange(
                indices,
                "(b h w) -> b h w",
                b=img_shape[0],
                h=latent_size,
                w=latent_size,
            )
        else:
            raise ValueError(f"Unsupported batch dimensions: {len(img_shape)}")

        return indices
        ############################################################

    @torch.no_grad()
    def tokenizer_decode_fn(indices, mini_bs=25):

        indices_shape = indices.shape
        if len(indices_shape) == 4:
            b, t, h, w = indices.shape
            indices = rearrange(indices, "b t h w -> (b t) (h w)")
        elif len(indices_shape) == 3:
            indices = rearrange(indices, "b h w -> b (h w)")
        else:
            raise ValueError(f"Unsupported batch dimensions: {len(indices_shape)}")
        # somelogic about video

        for i in range(0, len(indices), mini_bs):
            _indices = indices[i : i + mini_bs]
            _img = _tokenizer.decode_tokens(_indices)
            if i == 0:
                img = _img
            else:
                img = torch.cat([img, _img], dim=0)
        # somelogic about video
        if len(indices_shape) == 4:
            img = rearrange(img, "(b t) c h w -> b t c h w", b=b, t=t)

        img = img.clamp(-1, 1)
        img = ((img + 1) * 0.5 * 255.0).to(dtype=torch.uint8)
        return img

    return tokenizer_encode_fn, tokenizer_decode_fn


def extract_video():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ###########################################################
    avi_names = os.listdir(source_train)

    def video_generator():
        for _avi_id, _avi_name in tqdm(
            enumerate(avi_names),
            total=len(avi_names),
            desc="calculate total frames in dataset",
        ):
            _path = os.path.join(source_train, _avi_name)
            # Read the video file
            if is_debug and _avi_id > 10:
                break
            try:
                with open(_path, "rb") as stream:
                    video_data = stream.read()
                    vframes, aframes, info = torchvision.io.read_video(
                        io.BytesIO(video_data), pts_unit="sec", output_format="TCHW"
                    )
                    if len(vframes) < num_frames:
                        print(
                            f"video {_path} has less than {num_frames} frames, skipping"
                        )
                        continue

                    yield vframes
            except Exception as e:
                print(f"Error reading video file: {_path}")
                print(e)
                continue

    resize = transforms.Resize(resolution)

    if tokenizer_type == "sd_vq":
        tokenizer_encode_fn, tokenizer_decode_fn = SD_VQ_tokenizer()
    elif tokenizer_type == "titok":
        tokenizer_encode_fn, tokenizer_decode_fn = titok_tokenizer()
    else:
        raise ValueError(f"tokenizer={tokenizer_type} not supported")

    shutil.rmtree(fvd_video_dir, ignore_errors=True)
    os.makedirs(fvd_video_dir, exist_ok=True)
    video_gt_root_gt = os.path.join(fvd_video_dir, "gt")
    video_gt_root_reconstructed = os.path.join(fvd_video_dir, "reconstructed")
    os.makedirs(video_gt_root_gt, exist_ok=True)
    os.makedirs(video_gt_root_reconstructed, exist_ok=True)

    vg = video_generator()
    for _video_id, _vframes in enumerate(vg):
        vframes = resize(_vframes)
        vframes = vframes.numpy().astype(np.uint8)
        start_frame_ind, end_frame_ind = temporal_sample(len(vframes))
        assert end_frame_ind - start_frame_ind >= fvd_frames
        frame_indice = np.linspace(
            start_frame_ind, end_frame_ind - 1, fvd_frames, dtype=int
        )

        vframes = vframes[frame_indice]
        vframes = torch.from_numpy(vframes).to(device).float()
        reconstructed_vframes = tokenizer_decode_fn(tokenizer_encode_fn(vframes))
        video_gt_root = os.path.join(fvd_video_dir, "gt", f"{_video_id}")
        vframes = vframes.permute(0, 2, 3, 1)
        reconstructed_vframes = reconstructed_vframes.permute(0, 2, 3, 1)
        if True:
            torchvision.io.write_video(
                os.path.join(video_gt_root_gt, f"{_video_id}.mp4"),
                vframes,
                fps=write_fps,
            )
            torchvision.io.write_video(
                os.path.join(video_gt_root_reconstructed, f"{_video_id}.mp4"),
                reconstructed_vframes,
                fps=write_fps,
            )
        else:
            video_reconstructed_root = os.path.join(
                fvd_video_dir, "reconstructed", f"{_video_id}"
            )
            os.makedirs(video_gt_root, exist_ok=True)
            os.makedirs(video_reconstructed_root, exist_ok=True)
            vframes = vframes.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            reconstructed_vframes = (
                reconstructed_vframes.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            )
            assert vframes.shape == reconstructed_vframes.shape
            assert len(vframes) == fvd_frames
            print(vframes.shape)
            for i in range(len(vframes)):
                PIL.Image.fromarray(vframes[i]).save(
                    os.path.join(video_gt_root, f"{i}.png")
                )
                PIL.Image.fromarray(reconstructed_vframes[i]).save(
                    os.path.join(video_reconstructed_root, f"{i}.png")
                )

        if is_debug and _video_id > 40:
            break

        if _video_id > fvd_video_num:
            break


def calculate_fvd_stylegan(fake_root, real_root):

    print(real_root)
    print(fake_root)
    assert os.path.exists(real_root), f"real_root={real_root} not exists"
    assert os.path.exists(fake_root), f"fake_root={fake_root} not exists"
    print("video_num, real_root:", len(os.listdir(real_root)))
    print("video_num, fake_root:", len(os.listdir(fake_root)))

    calc_metrics_for_dataset(
        metrics=["fvd2048_16f"],
        real_data_path=real_root,
        fake_data_path=fake_root,
        mirror=True,
        resolution=256,
        gpus=1,
        verbose=False,
        use_cache=False,
        num_runs=1,
    )


if __name__ == "__main__":

    extract_video()
    if False:
        calculate_fvd_stylegan(
            fake_root=os.path.join(fvd_video_dir, "reconstructed"),
            real_root=os.path.join(fvd_video_dir, "gt"),
        )

    if False:
        calculate_fvd_github(
            gen_dir=os.path.join(
                "./data/unittest_ffs_tokenizer_titok_debug", "reconstructed"
            ),
            gt_dir=os.path.join("./data/unittest_ffs_tokenizer_titok_debug", "gt"),
            frames=16,
            resolution=64,
        )
    else:
        calculate_fvd_github(
            gen_dir=os.path.join(fvd_video_dir, "reconstructed"),
            gt_dir=os.path.join(fvd_video_dir, "gt"),
            frames=16,
            resolution=64,
        )
