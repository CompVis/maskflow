import sys

sys.path.append("..")

import os
import argparse
import numpy as np


from tqdm import tqdm
import torch
import os
import torch
import torchvision
import io
import h5py
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import transforms

#source_train = "~/data/preprocess_ffs/train/videos"
#source_train = os.path.expanduser(source_train)
#target_wds_dir = os.path.expanduser("./data/faceforensics_train_indices.h5")

source_train = "/dss/dsshome1/02/di38taq/droll/data/"
target_wds_dir = os.path.expanduser("./data/faceforensics_train_indices_32.h5")

resolution = 256
num_frames = 32
frame_interval = 3
is_debug = False
latent_size = 32

if is_debug:
    target_wds_dir = target_wds_dir.replace(".h5", "_debug.h5")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--split", type=str, default="train", help="split to convert"
    )
    parser.add_argument(
        "-c", "--category_name", type=str, default="cake", help="category name"
    )
    parser.add_argument("--max_size", type=float, default=0.1, help="gb per shard")
    opt = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ##################
    ldm_path = os.path.expanduser("/p/project/degeai/fuest1/droll/ldm")
    sys.path.insert(0, ldm_path)
    from ldm.util import instantiate_from_config

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
        indices = rearrange(
            indices,
            "(b h w) -> b h w",
            b=img_shape[0],
            h=latent_size,
            w=latent_size,
        )
        return indices
       

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

    frame_total = 0
    vg = video_generator()
    for _vframes in vg:
        frame_total += len(_vframes)
    print("frame_total: ", frame_total)
    h5_file = h5py.File(target_wds_dir, "w")
    h5_file.create_dataset(
        "video", (frame_total, latent_size, latent_size), dtype=np.int32
    )

    resize = transforms.Resize(resolution)
    start_index = 0
    start_index_list = []

    vg = video_generator()
    for _vframes in vg:
        vframes = resize(_vframes)
        vframes = vframes.numpy()
        vframes = vframes.astype(np.uint8)
        vframes = torch.from_numpy(vframes).to(device).float()
        indices = tokenizer_encode_fn(vframes)
        indices = indices.cpu().numpy()
        h5_file["video"][start_index : start_index + len(vframes)] = indices
        start_index_list.append(np.array([start_index, start_index + len(vframes)]))
        start_index += len(vframes)

    start_index_list = np.stack(start_index_list, axis=0)  # [N,2]
    h5_file.create_dataset("start_index_list", data=np.array(start_index_list))
    h5_file.close()
    print("done, saved to ", target_wds_dir)
