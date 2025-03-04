import os
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "2000"

import sys, math, uuid, time, random, shutil
import numpy as np, imageio
import torch, h5py, hydra, wandb
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator
from torchvision.io import write_video
from utils.train_utils import get_model, requires_grad
from utils_common import print_rank_0
from utils_vq import vq_get_dynamic

def load_sd_vq_f8_tokenizer(args, device):
    from ldm.util import instantiate_from_config
    sys.path.insert(0, os.path.abspath("./ldm"))
    config = OmegaConf.load(args.tokenizer.config_path)
    sd0 = torch.load(args.tokenizer.ckpt_path, map_location="cpu")["state_dict"]
    _tokenizer = instantiate_from_config(config.model)
    _tokenizer.load_state_dict(sd0, strict=False)
    _tokenizer.eval().requires_grad_(False).to(device)
    @torch.no_grad()
    def decode_fn(indices, mini_bs=25):
        if hasattr(args.tokenizer, "mask_token_id"):
            mid = getattr(args.tokenizer, "mask_token_id", None)
            if mid is not None:
                reid = getattr(args.tokenizer, "mask_token_reindex", mid)
                indices[indices == mid] = reid
        shp = indices.shape
        if len(shp) == 4:
            b, t, h, w = shp
            indices = rearrange(indices, "b t h w -> (b t) (h w)")
        elif len(shp) == 3:
            b, h, w = shp
            indices = rearrange(indices, "b h w -> b (h w)")
        else:
            raise ValueError("Unsupported shape")
        imgs = []
        for i in range(0, len(indices), mini_bs):
            chunk = indices[i:i+mini_bs].long()
            imgs.append(_tokenizer.decode_tokens(chunk))
        img = torch.cat(imgs, dim=0)
        if len(shp) == 4:
            b, t = shp[0], shp[1]
            img = rearrange(img, "(b t) c h w -> b t c h w", b=b, t=t)
        return ((img.clamp(-1,1)+1)*0.5*255).to(torch.uint8)
    return decode_fn

def get_context_frames_for_batch(args, batch_size, n_context, device):
    import h5py
    with h5py.File(args.data.h5_path, "r") as hf:
        start_idx_list = hf["start_index_list"][:]
    context_list = []
    for i in range(batch_size):
        vid_idx = (i + args.context_vid_offset) % len(start_idx_list)
        with h5py.File(args.data.h5_path, "r") as hf:
            s = int(hf["start_index_list"][vid_idx, 0])
            tokens = hf["video"][s:s+n_context]
        tokens = torch.from_numpy(tokens).to(device)
        context_list.append(tokens)
    return torch.stack(context_list, dim=0)

def make_stitched_evolution_grid(chain, decode_fn, device, outdir, context_frames, total_frames, num_spots=5, subsample=1, fps=2, sampler_name="maskgit"):
    chain_tensor = chain[0]  # expected shape: (L, B, T, H, W)
    L, B, T, H, W = chain_tensor.shape
    idx_list = np.linspace(context_frames - 1, total_frames - 1, num_spots).astype(int)
    chain_indices = list(range(0, L, subsample))
    if 0 not in chain_indices: chain_indices.append(0)
    if (L - 1) not in chain_indices: chain_indices.append(L - 1)
    chain_indices = sorted(set(chain_indices))
    grid_frames = []
    for l in chain_indices:
        state = chain_tensor[l]  # (B, T, H, W)
        decoded = decode_fn(state.to(device))  # (B, T, C, H, W)
        rows = []
        for b in range(B):
            video = decoded[b]  # (T, C, H, W)
            selected = []
            for fi in idx_list:
                if fi < video.shape[0]:
                    frame = video[fi]
                    frame_np = frame.cpu().numpy()
                    frame_np = np.transpose(frame_np, (1,2,0))
                    selected.append(frame_np)
            if selected:
                row = np.concatenate(selected, axis=1)
                rows.append(row)
        if rows:
            grid = np.concatenate(rows, axis=0)
            grid_frames.append(grid)
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"stitched_evolution_{sampler_name}.gif")
    imageio.mimsave(out_path, grid_frames, fps=fps, loop=0)
    print(f"Saved stitched gif to {out_path}")

@hydra.main(config_path="config", config_name="default", version_base=None)
def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index

    model = get_model(args)
    st = torch.load(args.ckpt, map_location=lambda s, l: s)
    md = {k.replace("module.",""): v for k, v in st["model"].items()}
    model.load_state_dict(md)
    model.to(device)
    requires_grad(model, False)
    model.eval()
    print_rank_0(f"Loaded checkpoint from {args.ckpt}")

    if args.tokenizer.name not in ["sd_vq_f8", "sd_vq_f8_size512"]:
        raise ValueError(f"Unsupported tokenizer: {args.tokenizer.name}")
    decode_fn = load_sd_vq_f8_tokenizer(args, device)
    _, sample_fn = vq_get_dynamic(args, device, is_train=False)

    def model_fn(*inp, **kw):
        if args.cfg_scale > 0 and hasattr(model, "forward_with_cfg"):
            return model.forward_with_cfg(*inp, cfg_scale=args.cfg_scale, **kw)
        return model.forward_without_cfg(*inp, **kw)

    outdir = os.path.join(args.sample_dir, f"sampling_chains_{args.dynamic.sampler}")
    os.makedirs(outdir, exist_ok=True)

    B = args.sampling_bs
    tot_frames = args.data.video_frames
    context_frames = args.dynamic.n_context_frames
    sample_size = (B, tot_frames, args.tokenizer.latent_size, args.tokenizer.latent_size)
    context = None
    if context_frames > 0:
        context = get_context_frames_for_batch(args, B, context_frames, device)

    with torch.no_grad():
        final_sample, chain = sample_fn(sample_size=sample_size, model=model_fn, context_indices=context, return_chains=True)
    make_stitched_evolution_grid(chain, decode_fn, device, outdir, context_frames, tot_frames,
                                 num_spots=args.gif_num_spots, subsample=args.gif_subsample, fps=args.fps,
                                 sampler_name=args.dynamic.sampler)
    print_rank_0("Done.")

if __name__=="__main__":
    main()
