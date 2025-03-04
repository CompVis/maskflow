import torch
import torch.nn.functional as F
import numpy as np
import uuid
import os
import shutil
from torchvision.utils import save_image
import click
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.fvd_calc_metrics_for_dataset import calc_metrics_for_dataset


class MyMetric_Offline_Video:
    def __init__(
        self,
        npz_real="./data/imagenet256_raw_wds_train_fidstat_real_50k.npz",
        device="cuda",
    ):
        fake_path = "./data/temp_fake"
        assert os.path.isfile(npz_real)
        self.npz_real = npz_real
        self.device = device
        self.fake = None
        self.fake_path = fake_path + "/" + str(uuid.uuid4().hex[:6])
        print("creating MyMetric_Offline fake path,", self.fake_path)
        shutil.rmtree(self.fake_path, ignore_errors=True)
        os.makedirs(self.fake_path)
        self.num_fake = 0

    def update_real(self, data, is_real=True):
        pass

    def update_fake(self, data, is_real=False):
        data = data.to(self.device)
        if len(data.shape) == 5:  # if it's a video, we evaluate the frame-level FID
            b, f, c, h, w = data.shape
            data = data.reshape(b * f, c, h, w)
        assert len(data.shape) == 4
        for _data in data:
            unique_id = uuid.uuid4().hex[:6]
            save_image(_data / 255.0, f"{self.fake_path}/{unique_id}.png")
        self.num_fake += len(data)

    def compute(self):
        print("computing metrics by npz file...")

        return dict()

    def reset(self):
        shutil.rmtree(self.fake_path, ignore_errors=True)
        os.makedirs(self.fake_path)
        self.num_fake = 0


@click.command()
def checkfid_video(
    root="~/lab/discretediffusion/samples/sample_vq_v2_note_ffs_dlatte_b2_0300000_bs4_fid50000_local",
):
    fake_root = os.path.expanduser(os.path.join(root, "samples"))
    real_root = os.path.expanduser(os.path.join(root, "gts"))
    print(real_root)
    print(fake_root)

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
    checkfid_video()
