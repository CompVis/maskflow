import sys

sys.path.append("..")

import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import h5py
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import transforms

# Update source and target directories for DMLab dataset
source_train = "/dss/dsshome1/02/di38taq/droll/data/dmlab"
target_wds_dir = os.path.expanduser("./data/dmlab_train_indices_36f.h5")

resolution = 256
num_frames = 36
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
        "-c", "--category_name", type=str, default="dmlab", help="category name"
    )
    parser.add_argument("--max_size", type=float, default=0.1, help="GB per shard")
    opt = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ##################
    # ldm_path = os.path.expanduser("lab/discretediffusion/ldm")
    sys.path.insert(0, '/dss/dsshome1/02/di38taq/droll/ldm')
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
        img_shape = img.shape

        indices_list = []
        for i in range(0, len(img), mini_bs):
            _img = img[i : i + mini_bs]
            encode_res = _tokenizer.encode(_img)
            _indices = encode_res[2][-1]
            indices_list.append(_indices)
        indices = torch.cat(indices_list, dim=0)
        indices = rearrange(
            indices,
            "(b h w) -> b h w",
            b=img_shape[0],
            h=latent_size,
            w=latent_size,
        )
        return indices
    

    ###########################################################
    # Get list of .npz files in the source directory
    npz_files = [f for f in os.listdir(source_train) if f.endswith('.npz')]

    def video_generator():
        for npz_file in tqdm(npz_files, desc="Processing npz files"):
            npz_path = os.path.join(source_train, npz_file)
            if is_debug and npz_files.index(npz_file) > 10:
                break
            try:
                data = np.load(npz_path)

                for key in data.files:
                    if key == "actions":
                        continue
                    vframes = data[key]  # Shape: [num_frames, H, W, C]
                    if vframes.ndim != 4 or vframes.shape[-1] != 3:
                        print(f"Skipping {key} in {npz_file}, unexpected shape {vframes.shape}")
                        continue
                    vframes = torch.from_numpy(vframes).permute(0, 3, 1, 2)  # Convert to [T, C, H, W]
                    if len(vframes) < num_frames:
                        print(
                            f"Video {key} in {npz_file} has less than {num_frames} frames, skipping"
                        )
                        continue
                    yield vframes
            except Exception as e:
                print(f"Error reading npz file: {npz_path}")
                print(e)
                continue

    # Calculate total number of frames
    frame_total = 0
    vg = video_generator()
    for _vframes in vg:
        print(len(_vframes))
        frame_total += len(_vframes)
    print("Total frames: ", frame_total)

    # Create H5 file
    h5_file = h5py.File(target_wds_dir, "w")
    h5_file.create_dataset(
        "video", (frame_total, latent_size, latent_size), dtype=np.int32
    )

    resize = transforms.Resize(resolution)
    start_index = 0
    start_index_list = []

    vg = video_generator()
    resize_to_64 = transforms.Resize(resolution)
    upsample_to_256 = transforms.Resize(256)

    vg = video_generator()
    for _vframes in vg:
        # Step 1: Resize to 64 resolution
        vframes_resized = resize_to_64(_vframes)  # [T, C, 64, 64]

        # Step 2: Upsample to 256 resolution
        vframes_upsampled = upsample_to_256(vframes_resized)  # [T, C, 256, 256]

        # Step 3: Convert to numpy and ensure proper format
        vframes_upsampled = vframes_upsampled.numpy()
        vframes_upsampled = vframes_upsampled.astype(np.uint8)

        # Step 4: Encode the upsampled frames using the tokenizer
        vframes_tensor = torch.from_numpy(vframes_upsampled).to(device).float()
        indices = tokenizer_encode_fn(vframes_tensor)

        # Step 5: Save the indices to the H5 dataset
        indices = indices.cpu().numpy()
        h5_file["video"][start_index : start_index + len(vframes_tensor)] = indices
        start_index_list.append(np.array([start_index, start_index + len(vframes_tensor)]))
        start_index += len(vframes_tensor)

    start_index_list = np.stack(start_index_list, axis=0)  # [N, 2]
    h5_file.create_dataset("start_index_list", data=start_index_list)
    h5_file.close()
    print("Done, saved to ", target_wds_dir)
