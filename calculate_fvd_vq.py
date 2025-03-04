import os
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "2000"

import sys
import math
import uuid
import time
import torch
import h5py
import random
import hydra
import wandb
import shutil
import numpy as np
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from accelerate import Accelerator
from torchvision.io import write_video
from utils.train_utils import get_model, requires_grad
from utils_common import print_rank_0
from utils_vq import vq_get_dynamic, vq_get_vae
from fvd_external import calculate_fvd_github

def get_random_consecutive_frames_from_h5(h5_path, index, num_frames, device, start_index_override=None):
    """
    Loads a random (or fixed) consecutive segment of length `num_frames` from 
    the H5 file for the specified video index. If `start_index_override` is given, 
    use that instead of a random offset.
    """
    with h5py.File(h5_path, "r") as hf:
        start_ = int(hf['start_index_list'][index, 0])
        end_   = int(hf['start_index_list'][index, 1])
        total_length = end_ - start_
        if total_length < num_frames:
            raise ValueError(f"Video {index} has only {total_length} frames.")

        if start_index_override is not None:
            offset = start_index_override
            offset = min(max(offset, 0), total_length - num_frames)
        else:
            offset = random.randint(0, total_length - num_frames)

        chosen_start = start_ + offset
        chosen_end = chosen_start + num_frames
        video_tokens = hf['video'][chosen_start:chosen_end]
        return torch.from_numpy(video_tokens).unsqueeze(0).to(device)


def load_sd_vq_f8_tokenizer(args, device):
    from ldm.util import instantiate_from_config
    sys.path.insert(0, os.path.abspath("./ldm"))
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
            chunk = indices[start_idx : start_idx + mini_bs].long()
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

def decode_faceforensics_h5_to_mp4_parallel(
    h5_path: str,
    output_dir: str,
    tokenizer_decode_fn,
    device,
    frames: int,
    max_videos: int,
    rank: int,
    world_size: int,
):
    """
    Each rank decodes a subset of videos.
    """
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as hf:
        start_indices = hf["start_index_list"][()]
        total_videos = min(len(start_indices), max_videos)

    per_rank = math.ceil(total_videos / world_size)
    start_idx = rank * per_rank
    end_idx = min(start_idx + per_rank, total_videos)

    for i in tqdm(range(start_idx, end_idx), desc=f"Decoding on rank {rank}"):
        try:
            video_tokens = get_random_consecutive_frames_from_h5(h5_path, i, frames, device)
            decoded_frames = tokenizer_decode_fn(video_tokens).squeeze(0).clamp(0, 255).byte()
            if decoded_frames.shape[0] < frames:
                print(f"Rank {rank}: Video {i} has insufficient frames. Skipping.")
                continue
            decoded_frames = rearrange(decoded_frames, "t c h w -> t h w c").cpu().numpy()
            unique_id = f"video_{i}_rank{rank}_{uuid.uuid4().hex[:4]}"
            mp4_path = os.path.join(output_dir, f"{unique_id}.mp4")
            write_video(mp4_path, decoded_frames, fps=8)
        except Exception as e:
            print(f"Rank {rank}: Error processing video {i}: {e}")
            continue

@hydra.main(config_path="config", config_name="default", version_base=None)
def main(args):
    print_rank_0(f"NCCL_ASYNC_ERROR_HANDLING={os.environ.get('NCCL_ASYNC_ERROR_HANDLING')}")
    print_rank_0(f"NCCL_BLOCKING_WAIT={os.environ.get('NCCL_BLOCKING_WAIT')}")
    print_rank_0(f"NCCL_TIMEOUT={os.environ.get('NCCL_TIMEOUT')}")

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is None:
        slurm_job_id = "local"
    print_rank_0(f"slurm_job_id: {slurm_job_id}")

    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    wandb_run_name = (
        f"fvd_eval_{args.data.name}_"
        f"{os.path.basename(args.ckpt)}_"
        f"{args.dynamic.sampler}_"
        f"{args.dynamic.sampling_horizon}_frames"
        f"{args.dynamic.n_context_frames}_"
        f"{slurm_job_id}"
    )

    if rank == 0 and getattr(args, "use_wandb", False):
        wandb.init(
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=OmegaConf.to_container(args, resolve=True),
            name=wandb_run_name,
            dir="./wandb",
        )

    model = get_model(args)
    state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    _model_dict = state_dict["model"]
    print(f"_model_dict keys: {_model_dict.keys()}")
    _model_dict = {k.replace("module.", ""): v for k, v in _model_dict.items()}
    model.load_state_dict(_model_dict)
    model.to(device)
    requires_grad(model, False)
    model.eval()
    print_rank_0(f"Loaded checkpoint from {args.ckpt}")

    if args.tokenizer.name not in ["sd_vq_f8", "sd_vq_f8_size512"]:
        raise ValueError(f"Unsupported tokenizer name: {args.tokenizer.name}")
    tokenizer_decode_fn = load_sd_vq_f8_tokenizer(args, device)

    real_videos_dir = f"real_videos_{args.data.name}_{args.gt_video_frames}"
    real_videos_dir = os.path.join(args.sample_dir, real_videos_dir)

    accelerator.wait_for_everyone()

    existing_videos = []
    if os.path.isdir(real_videos_dir):
        existing_videos = [f for f in os.listdir(real_videos_dir) if f.endswith(".mp4")]

    if len(existing_videos) < args.num_fid_samples:
        if os.path.exists(real_videos_dir) and rank == 0:
            shutil.rmtree(real_videos_dir)
        accelerator.wait_for_everyone()
        os.makedirs(real_videos_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        decode_faceforensics_h5_to_mp4_parallel(
            h5_path=args.data.h5_path,
            output_dir=real_videos_dir,
            tokenizer_decode_fn=tokenizer_decode_fn,
            device=device,
            frames=args.gt_video_frames,
            max_videos=args.num_fid_samples,
            rank=rank,
            world_size=world_size,
        )
        accelerator.wait_for_everyone()

    os.makedirs(real_videos_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    real_video_files = sorted([
        f for f in os.listdir(real_videos_dir)
        if f.endswith(".mp4")
    ])[:args.num_fid_samples]

    per_gpu = math.ceil(len(real_video_files) / world_size)
    my_real_videos = real_video_files[rank * per_gpu : (rank + 1) * per_gpu]
    training_losses_fn, sample_fn = vq_get_dynamic(args, device, is_train=False)

    def model_fn(*inputs, **kwargs):
            return model.forward_without_cfg(*inputs, **kwargs)
    
    if hasattr(args, "use_existing_sample_dir"):
        gen_videos_dir = os.path.join(args.sample_dir, args.use_existing_sample_dir)
        if not os.path.isdir(gen_videos_dir):
            raise ValueError(f"Prepared gen_videos_dir does not exist: {gen_videos_dir}")

        print_rank_0(f"Skipping generation. Using existing sample videos at {gen_videos_dir}.")
    else:
        gen_videos_dir = (
            f"gen_videos_{args.data.name}_"
            f"{os.path.basename(args.ckpt)}_"
            f"{slurm_job_id}_"
            f"{args.dynamic.sampler}_{args.dynamic.sampling_timesteps}_"
            f"{args.dynamic.sampling_horizon}_frames_"
            f"context_{args.dynamic.n_context_frames}_"
            f"stride_{args.dynamic.sampling_window_stride}_"
        )
        gen_videos_dir = os.path.join(args.sample_dir, gen_videos_dir)

        if rank == 0:
            if os.path.exists(gen_videos_dir):
                shutil.rmtree(gen_videos_dir)
        accelerator.wait_for_everyone()

        # Ensure the gen_videos_dir is created on all ranks
        os.makedirs(gen_videos_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        # times = []
        # loop_iter = range(len(my_real_videos))
        # pbar = tqdm(loop_iter, desc="Generating Fake Videos") if rank == 0 else loop_iter

        my_indices = list(range(len(my_real_videos)))
        loop_iter = range(0, len(my_indices), args.data.sample_fid_bs)
        pbar = tqdm(loop_iter, desc="Generating Fake Videos", disable=(rank != 0))
        times = []

        for chunk_start in pbar:
            start_time = time.time()

            # Slice the next chunk of indices
            chunk_end = min(chunk_start + args.data.sample_fid_bs, len(my_indices))
            batch_indices = my_indices[chunk_start:chunk_end]
            B = len(batch_indices)  # actual batch size (could be smaller at the end)

            if B == 0:
                continue

            if args.dynamic.n_context_frames > 0:
                batch_context_list = []
                for i_local in batch_indices:
                    video_idx = i_local + rank * per_gpu
                    context_override = None
                    
                    full_context = get_random_consecutive_frames_from_h5(
                        args.data.h5_path,
                        video_idx,
                        args.data.video_frames,
                        device,
                        start_index_override=context_override
                    )
                    
                    context = full_context[:, :args.dynamic.n_context_frames]
                    batch_context_list.append(context)

                context_indices = torch.cat(batch_context_list, dim=0)
            else:
                context_indices = None


            sample_size = (
                B,
                args.data.video_frames,
                args.tokenizer.latent_size,
                args.tokenizer.latent_size,
            )
            
            with torch.no_grad():
            
                chains = sample_fn(
                    sample_size=sample_size,
                    model=model_fn,      
                    context_indices=context_indices,
                )

            codes = chains[-1]
            decoded = tokenizer_decode_fn(codes)

            for i_in_batch, real_idx in enumerate(batch_indices):
                out_video = decoded[i_in_batch]  
                out_video = rearrange(out_video, "t c h w -> t h w c").cpu().numpy()

                unique_id = f"rank{rank}_vid{real_idx}_{uuid.uuid4().hex[:4]}"
                mp4_path = os.path.join(gen_videos_dir, f"{unique_id}.mp4")
                write_video(mp4_path, out_video, fps=8)

            end_time = time.time()
            times.append(end_time - start_time)

        partial_time = torch.tensor(sum(times), device=device)
        partial_count = torch.tensor(len(times), device=device)
        total_time = accelerator.reduce(partial_time, reduction="sum")
        total_count = accelerator.reduce(partial_count, reduction="sum")

        if rank == 0 and total_count.item() > 0:
            avg_time = (total_time / total_count).item()
            print_rank_0(f"Average sampling time (seconds): {avg_time:.4f}")
            if getattr(args, "use_wandb", False):
                wandb.log({"avg_sampling_time": avg_time})

    accelerator.wait_for_everyone()

    if rank == 0:
        print_rank_0("Calculating FVD...")
        gen_video_files = sorted([f for f in os.listdir(gen_videos_dir) if f.endswith(".mp4")])
        real_video_files = sorted([f for f in os.listdir(real_videos_dir) if f.endswith(".mp4")])

        final_count = min(len(gen_video_files), len(real_video_files))
        gen_video_files = gen_video_files[:final_count]
        real_video_files = real_video_files[:final_count]

        temp_real_dir = os.path.join(args.sample_dir, f"temp_real_videos_{slurm_job_id}")
        temp_gen_dir = os.path.join(args.sample_dir, f"temp_gen_videos_{slurm_job_id}")
        
        
        if os.path.exists(temp_real_dir):
            shutil.rmtree(temp_real_dir)
        if os.path.exists(temp_gen_dir):
            shutil.rmtree(temp_gen_dir)

        os.makedirs(temp_real_dir, exist_ok=True)
        os.makedirs(temp_gen_dir, exist_ok=True)

        for video in real_video_files:
            shutil.copy(os.path.join(real_videos_dir, video), temp_real_dir)
        for video in gen_video_files:
            shutil.copy(os.path.join(gen_videos_dir, video), temp_gen_dir)

        print_rank_0(f"Final # real videos: {len(real_video_files)}")
        print_rank_0(f"Final # generated videos: {len(gen_video_files)}")

        results = calculate_fvd_github(
            gen_dir=temp_gen_dir,
            gt_dir=temp_real_dir,
            resolution=args.data.image_size,
            frames=args.data.video_frames,
            sampling=args.sampling,
        )
        print_rank_0("FVD Results:", results)
        if getattr(args, "use_wandb", False):
            wandb.log({"FVD": results.get("fvd", None)})

        if getattr(args, "remove_videos_after", False):
            for d in [temp_real_dir, temp_gen_dir, gen_videos_dir]:
                shutil.rmtree(d, ignore_errors=True)
            print_rank_0("Removed video directories.")

    print_rank_0("Sampling and FVD calculation completed.")


if __name__ == "__main__":
    main()