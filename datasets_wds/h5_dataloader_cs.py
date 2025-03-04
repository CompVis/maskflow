"""This file contains the definition of data loader using webdataset.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    https://github.com/huggingface/open-muse/blob/main/training/data.py
"""

import math
from typing import List, Union, Text
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from einops import rearrange
import torch
import h5py
from torch.utils.data import DataLoader
import random

import numpy as np
import io
from torchvision.transforms import functional as F


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


class CROP_IMG_SEGMASK:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, label):
        crop_x, crop_y = random.randint(
            0, img.shape[1] - self.crop_size
        ), random.randint(0, img.shape[2] - self.crop_size)
        return (
            img[:, crop_x : crop_x + self.crop_size, crop_y : crop_y + self.crop_size],
            label[
                :, crop_x : crop_x + self.crop_size, crop_y : crop_y + self.crop_size
            ],
        )


class H5_CS_Dataset(Dataset):
    def __init__(
        self,
        h5_path,
        image_size: int,
        per_gpu_batch_size=4,
        num_workers_per_gpu=2,
        **kwargs,
    ):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, "r") as f:
            self.len = len(f["images"])
            print("cityscapes dataset len:", self.len)

        self.random_crop = CROP_IMG_SEGMASK(image_size)
        self.image_size = image_size

    def __len__(self):
        return self.len

    def _open_hdf5(self):
        self._hf = h5py.File(self.h5_path, "r")

    def __getitem__(self, index):
        if not hasattr(self, "_hf"):
            self._open_hdf5()

        img = np.array(self._hf["images"][index])
        label = np.array(self._hf["labels"][index])
        img, label = torch.tensor(img), torch.tensor(label)
        img = img / 255.0
        img = img * 2 - 1
        img, label = self.random_crop(img, label)

        label = (
            F.resize(
                label.unsqueeze(0),
                (self.image_size // 8, self.image_size // 8),
                interpolation=F.InterpolationMode.NEAREST,
            )
            .squeeze(0)
            .squeeze(0)
        ).long()  # as for Stable Diffusion, it's downsample to 1/8
        return dict(image=img, cls_id=label)  # [3,W,H],[W/8,H/8]


def get_inf_h5_dataloader(**kwargs):
    per_gpu_batch_size = kwargs["per_gpu_batch_size"]
    num_workers_per_gpu = kwargs["num_workers_per_gpu"]
    dataloader = H5_CS_Dataset(**kwargs)

    def _gen():
        dl = DataLoader(
            dataloader,
            batch_size=per_gpu_batch_size,
            num_workers=num_workers_per_gpu,
            shuffle=True,
        )
        while True:
            for batch in dl:
                yield batch

    return _gen()


if __name__ == "__main__":

    if False:
        dataloader = H5_CS_Dataset(
            h5_path="./data/cityscapes_res256.h5",
            image_size=256,
        )

        dl = DataLoader(dataloader, batch_size=4, num_workers=1)

        for batch in dl:
            print(batch.keys())
            print(
                batch["image"].shape,
                batch["image"].max(),
                batch["image"].min(),
                batch["image"].unique(),
            )
            print(
                batch["cls_id"].shape,
                batch["cls_id"].max(),
                batch["cls_id"].min(),
                batch["cls_id"].unique(),
            )
            break
    else:
        dataloader = get_inf_h5_dataloader(
            h5_path="./data/cityscapes_res256.h5",
            image_size=256,
            per_gpu_batch_size=40,
            num_workers_per_gpu=1,
        )

        for batch in dataloader:
            if False:
                print(batch.keys())
                print(
                    batch["image"].shape,
                    batch["image"].max(),
                    batch["image"].min(),
                )
                print(
                    batch["cls_id"].shape,
                    batch["cls_id"].max(),
                    batch["cls_id"].min(),
                    batch["cls_id"].unique(),
                )
            break
