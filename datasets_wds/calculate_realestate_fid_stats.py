import os
import h5py
import numpy as np
import cv2
import tempfile
import shutil
import argparse
from tqdm import tqdm
import torch
from omegaconf import OmegaConf
from einops import rearrange
import sys
from accelerate import Accelerator


sys.path.insert(0, os.path.abspath("./ldm"))
sys.path.insert(0, '/dss/dsshome1/02/di38taq/droll/')
from ldm.util import instantiate_from_config
from utils.eval_tools.fid_score import calculate_real_statistics

def load_sd_vq_f8_tokenizer(args, device):
    config = OmegaConf.load(args.tokenizer.config_path)
    pl_sd = torch.load(args.tokenizer.ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"]
    _tokenizer = instantiate_from_config(config.model)
    _tokenizer.load_state_dict(sd, strict=False)
    _tokenizer.eval()
    _tokenizer.requires_grad_(False)
    _tokenizer = _tokenizer.to(device)

    @torch.no_grad()
    def tokenizer_decode_fn(indices, mini_bs=25):
        indices = indices.to(device)
        if hasattr(args.tokenizer, "mask_token_id"):
            mask_id = getattr(args.tokenizer, "mask_token_id", None)
            if mask_id is not None:
                reindex_id = getattr(args.tokenizer, "mask_token_reindex", mask_id)
                indices[indices == mask_id] = reindex_id

        shape_ = indices.shape
        if len(shape_) == 4:
            b, t, h, w = shape_
            indices = rearrange(indices, "b t h w -> (b t) (h w)")
        elif len(shape_) == 3:
            b, h, w = shape_
            indices = rearrange(indices, "b h w -> b (h w)")
        else:
            raise ValueError("Unsupported indices shape for decode")

        imgs = []
        start_idx = 0
        while start_idx < len(indices):
            chunk = indices[start_idx : start_idx + mini_bs].long().to(device)
            chunk_img = _tokenizer.decode_tokens(chunk)
            imgs.append(chunk_img)
            start_idx += mini_bs
        img = torch.cat(imgs, dim=0)
        if len(shape_) == 4:
            img = rearrange(img, "(b t) c h w -> b t c h w", b=b, t=t)
        img = img.clamp(-1, 1)
        img = ((img + 1) * 0.5 * 255.0).to(torch.uint8)
        return img

    return tokenizer_decode_fn

def gen_fid_stats_realestate_npz(args, tokenizer_decode_fn, fid_num, output_npz_path, device="cuda:0"):
    temp_dir = tempfile.mkdtemp(prefix="realestate_decoded_")
    print(f"Decoding frames and saving images to temporary directory: {temp_dir}")
    with h5py.File(args.h5_path, "r") as f:
        if "data" not in f:
            print("Dataset 'data' not found in the h5 file.")
            return
        tokens_data = f["data"][:]
    total_frames = tokens_data.shape[0]
    print(f"Total frames in tokenized h5 file: {total_frames}")
    count = 0
    for i in tqdm(range(total_frames), desc="Decoding frames"):
        if count >= fid_num:
            break
        tokens = tokens_data[i]
        
        if tokens.ndim == 1:
            n = tokens.shape[0]
            side = int(np.sqrt(n))
            tokens = tokens.reshape(1, side, side)
        elif tokens.ndim == 2:
            tokens = tokens[None, ...]
        
        tokens_tensor = torch.tensor(tokens, device=device)
        img_tensor = tokenizer_decode_fn(tokens_tensor)  
        if img_tensor.ndim == 4 and img_tensor.shape[0] == 1:
            img_tensor = img_tensor.squeeze(0)
        img = img_tensor.cpu().permute(1, 2, 0).numpy()  
        out_path = os.path.join(temp_dir, f"{i}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        count += 1
    calculate_real_statistics(path_real=temp_dir, out_path_npy=output_npz_path, device=device)
    print(f"FID statistics saved to {output_npz_path}")
    shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, default="./data/realestate_10k_indices_1000.h5",
                        help="Path to the tokenized h5 file")
    parser.add_argument("--fid_num", type=int, default=50000,
                        help="Number of frames to decode for FID computation")
    parser.add_argument("--output_npz", type=str,
                        default="/dss/dsshome1/02/di38taq/droll/data/realestate_10k_fid_stats.npz",
                        help="Output path for FID stats in npz format")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (e.g., cuda:0)")
    
    parser.add_argument("--tokenizer_config_path", type=str,
                        default="./ldm/models/first_stage_models/vq-f8/config.yaml",
                        help="Path to tokenizer config file")
    parser.add_argument("--tokenizer_ckpt_path", type=str,
                        default="./pretrained_ckpt/ldm/vq-f8.ckpt",
                        help="Path to tokenizer checkpoint file")
    args = parser.parse_args()
    
    class TokenizerArgs:
        pass
    args.tokenizer = TokenizerArgs()
    args.tokenizer.config_path = args.tokenizer_config_path
    args.tokenizer.ckpt_path = args.tokenizer_ckpt_path

    accelerator = Accelerator()
    device = accelerator.device
    args.device = device
    print(f"Using device: {device}")

    tokenizer_decode_fn = load_sd_vq_f8_tokenizer(args, device)
    gen_fid_stats_realestate_npz(args, tokenizer_decode_fn, args.fid_num, args.output_npz, device=device)

if __name__ == "__main__":
    main()
