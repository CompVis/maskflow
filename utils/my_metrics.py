from einops import rearrange
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from utils.torchmetric_fdd import FrechetDinovDistance
from utils.torchmetric_fvd import FrechetVideoDistance
from utils.torchmetric_prdc import PRDC
from utils.torchmetric_sfid import sFrechetInceptionDistance
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


class MyMetric:
    def __init__(
        self, device="cuda", choices=["fid"], video_frame=None, sync_on_compute=True
    ):
        self.choices = choices
        self.device = device
        if "fid" in choices:
            self._fid = FrechetInceptionDistance(
                feature=2048,
                reset_real_features=True,
                normalize=False,
                sync_on_compute=sync_on_compute,
            ).to(device)
        if "is" in choices:
            self._is = InceptionScore(
                sync_on_compute=sync_on_compute,
            ).to(device)
        if "kid" in choices:
            self._kid = KernelInceptionDistance(
                sync_on_compute=sync_on_compute, subset_size=50
            ).to(device)
        if "prdc" in choices:
            self._prdc = PRDC(sync_on_compute=sync_on_compute, nearest_k=5).to(device)
        if "sfid" in choices:
            self._sfid = sFrechetInceptionDistance(
                sync_on_compute=sync_on_compute,
            ).to(device)
        if "fdd" in choices:
            self._fdd = FrechetDinovDistance(
                sync_on_compute=sync_on_compute,
            ).to(device)
        if "fvd" in choices:
            self._fvd = FrechetVideoDistance()
            self.video_frame = video_frame
            assert video_frame is not None, "video_frame is None"
        if "dinov2" in choices:
            from utils.torchmetric_dinov2 import DinoV2_Metric

            self._dinov2 = DinoV2_Metric().to(device)

    def update_real(self, data, is_real=True):
        data = data.to(self.device)
        self.update_fake_and_real(data, is_real)

    def update_fake(self, data, is_real=False):
        data = data.to(self.device)
        if "is" in self.choices:
            self._is.update(data)
        self.update_fake_and_real(data, is_real)

    def update_fake_and_real(self, data, is_real):
        if "fid" in self.choices:
            self._fid.update(data, real=is_real)
        if "kid" in self.choices:
            self._kid.update(data, real=is_real)
        if "prdc" in self.choices:
            self._prdc.update(data, real=is_real)
        if "sfid" in self.choices:
            self._sfid.update(data, real=is_real)
        if "fdd" in self.choices:
            self._fdd.update(data, real=is_real)
        if "fvd" in self.choices:
            assert isinstance(data, torch.Tensor) and data.dtype == torch.uint8
            # data is a torch.Tensor of type uint8
            # data = (rearrange(data, "b t c h w -> b t h w c") / 255.0 - 0.5) * 2
            data = rearrange(data, "(b t) c h w -> b t c h w", t=self.video_frame)
            b, t, c, h, w = data.shape
            data = rearrange(data, "b t c h w -> (b t) c h w").float()
            data = F.interpolate(
                data, size=(224, 224), mode="bilinear", align_corners=False
            )
            data = rearrange(data, "(b t) c h w -> b t h w c", t=t).float()
            self._fvd.update(data.to(self.device), real=is_real)
        if "dinov2" in self.choices:
            self._dinov2.update(data, real=is_real)

    def compute(self):
        print("computing torchmetrics...")
        _result = dict()
        if "fid" in self.choices:
            fid = self._fid.compute().item()
            _result["num_real"] = self._fid.real_features_num_samples
            _result["num_fake"] = self._fid.fake_features_num_samples
            _result["fid"] = fid
        if "is" in self.choices:
            _is_mean, _is_std = self._is.compute()
            _result["is"] = _is_mean.item()
        if "kid" in self.choices:
            _kid_mean, _kid_std = self._kid.compute()
            _result["kid_mean"] = _kid_mean.item()
            _result["kid_std"] = _kid_std.item()
        if "prdc" in self.choices:
            _prdc_result = self._prdc.compute()
            _prdc_result = {f"prdc_{k}": v for k, v in _prdc_result.items()}
            _result.update(_prdc_result)
        if "sfid" in self.choices:
            sfid = self._sfid.compute().item()
            _result["sfid"] = sfid
        if "fdd" in self.choices:
            fdd = self._fdd.compute().item()
            _result["fdd"] = fdd
        if "fvd" in self.choices:
            fvd = self._fvd.compute().item()
            _result["fvd"] = fvd
        if "dinov2" in self.choices:
            dinov2_dict = self._dinov2.compute()
            _result.update(dinov2_dict)
        _result = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in _result.items()
        }
        _result = {
            k: float(v) if isinstance(v, np.float64) else v for k, v in _result.items()
        }
        return _result

    def reset(self):
        if "fid" in self.choices:
            self._fid.reset()
        if "is" in self.choices:
            self._is.reset()
        if "kid" in self.choices:
            self._kid.reset()
        if "prdc" in self.choices:
            self._prdc.reset()
        if "sfid" in self.choices:
            self._sfid.reset()
        if "fdd" in self.choices:
            self._fdd.reset()
        if "fvd" in self.choices:
            self._fvd.reset()
        if "dinov2" in self.choices:
            self._dinov2.reset()


import os
from PIL import Image


class DirectoryDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]
        for _img_path in self.img_paths:
            try:
                _img = Image.open(_img_path)
            except Exception as e:
                print(f"Error opening image {_img_path}: {e}")
                self.img_paths.remove(_img_path)
        print(f"Found {len(self.img_paths)} valid images in {self.root_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        _img = Image.open(img_path)
        _img = torch.from_numpy(np.array(_img))
        _img = rearrange(_img, "h w c -> c h w")
        return _img


def calculate_fid_given_paths_my_metric(
    sample_img_dir, gt_img_dir, batch_size=50, num_workers=8
):
    _metric = MyMetric(
        choices=[
            "fid",
            "is",
            "kid",
            "prdc",
            "sfid",
        ],
        sync_on_compute=False,  # i will guarantee I only run it on single gpu
    )
    sample_imgs = os.listdir(sample_img_dir)
    gt_imgs = os.listdir(gt_img_dir)
    sample_dataset = DirectoryDataset(sample_img_dir)
    gt_dataset = DirectoryDataset(gt_img_dir)
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    gt_dataloader = DataLoader(
        gt_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("sample_imgs", len(sample_imgs), "gt_imgs", len(gt_imgs))
    for _img in tqdm(sample_dataloader, desc="sample_imgs"):
        _metric.update_fake(_img)
    for _img in tqdm(gt_dataloader, desc="gt_imgs"):
        _metric.update_real(_img)
    return _metric.compute()


if __name__ == "__main__":
    _metric = MyMetric(
        choices=["fid", "is", "kid", "prdc", "sfid", "fdd"],
    )
    _metric.update_real(
        torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8).to("cuda")
    )
    _metric.update_fake(
        torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8).to("cuda")
    )

    print(_metric.compute())
