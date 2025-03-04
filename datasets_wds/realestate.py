import sys
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
import cv2
import wandb
from accelerate import Accelerator

DEFAULT_SOURCE_DIR = "/dss/mcmlscratch/02/di38taq/real-state-10k/downloaded"
DEFAULT_H5_PATH = "/dss/mcmlscratch/02/di38taq/realestate10k_indices_1000_2.h5"
DEFAULT_RESOLUTION = 256
LATENT_SIZE = 32

def is_extensionless_or_webm(fname: str):
    if '.' not in fname:
        return True
    ext = os.path.splitext(fname)[1].lower()
    if ext == ".webm":
        return True
    return False

def load_video_frames(filepath, resolution=256):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != resolution or frame.shape[1] != resolution:
            frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return None
    frames = np.stack(frames, axis=0)
    return frames

def save_debug_mp4(frames_rgb, out_path="debug_clip.mp4", fps=10):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frames_rgb.shape[1], frames_rgb.shape[2]
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for frame in frames_rgb:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()

def tokenizer_encode_fn(img, tokenizer, mini_bs=4):
    img = img / 255.0
    img = (img - 0.5) * 2
    B = img.shape[0]
    indices_list = []
    for i in range(0, B, mini_bs):
        _img = img[i : i + mini_bs]
        encode_res = tokenizer.encode(_img)
        _indices = encode_res[2][-1]
        indices_list.append(_indices)
    indices = torch.cat(indices_list, dim=0)
    indices = rearrange(indices, "(b h w) -> b h w", b=B, h=LATENT_SIZE, w=LATENT_SIZE)
    return indices

def load_file_list(source_dir, max_videos):
    allfiles = os.listdir(source_dir)
    allfiles = [os.path.join(source_dir, f) for f in allfiles if is_extensionless_or_webm(f)]
    allfiles = sorted(allfiles)
    if len(allfiles) > max_videos:
        allfiles = allfiles[:max_videos]
    return allfiles

def process_videos(filepaths, device, tokenizer, args, h5_out_path):
    total_frames = 0
    metadata = []
    for filepath in tqdm(filepaths, desc="Counting frames"):
        frames_np = load_video_frames(filepath, resolution=args.resolution)
        if frames_np is None:
            continue
        num_frames = frames_np.shape[0]
        total_frames += num_frames
        metadata.append((filepath, num_frames))
    h5_file = h5py.File(h5_out_path, "w")
    dset_video = h5_file.create_dataset("video", (total_frames, LATENT_SIZE, LATENT_SIZE), dtype="int32")
    start_index_list = []
    running_index = 0
    saved_debug_mp4 = False
    for i, (filepath, num_frames) in enumerate(tqdm(metadata, desc="Tokenizing & writing H5")):
        frames_np = load_video_frames(filepath, resolution=args.resolution)
        if frames_np is None or frames_np.shape[0] != num_frames:
            continue
        if not saved_debug_mp4:
            debug_clip = frames_np[: min(50, num_frames)]
            save_debug_mp4(debug_clip, out_path=args.debug_video_path, fps=10)
            saved_debug_mp4 = True
            if args.use_wandb:
                wandb.log({"debug_clip_saved": True, "debug_video_path": args.debug_video_path})
        frames_tensor_cpu = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
        indices_list = []
        chunk_size = 16
        for start in range(0, num_frames, chunk_size):
            end = start + chunk_size
            chunk_cpu = frames_tensor_cpu[start:end]
            chunk_gpu = chunk_cpu.to(device)
            with torch.no_grad():
                chunk_indices = tokenizer_encode_fn(chunk_gpu, tokenizer, mini_bs=1)
            indices_list.append(chunk_indices.cpu())
            del chunk_gpu, chunk_indices
            torch.cuda.empty_cache()
        indices = torch.cat(indices_list, dim=0).numpy()
        dset_video[running_index : running_index + num_frames] = indices
        start_index_list.append([running_index, running_index + num_frames])
        running_index += num_frames
        if args.use_wandb:
            wandb.log({"videos_processed": i + 1, "videos_remaining": len(metadata) - (i + 1)})
    start_index_list = np.array(start_index_list, dtype=np.int32)
    h5_file.create_dataset("start_index_list", data=start_index_list)
    h5_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_h5", type=str, default=DEFAULT_H5_PATH)
    parser.add_argument("--max_videos", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--debug_video_path", type=str, default="debug_clip.mp4")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rdm")
    parser.add_argument("--wandb_run_name", type=str, default="realestate10k_h5_writer")
    parser.add_argument("--ckpt_path", type=str, default="./pretrained_ckpt/ldm/vq-f8.ckpt")
    parser.add_argument("--config_path", type=str, default="./ldm/models/first_stage_models/vq-f8/config.yaml")
    args = parser.parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=f"{args.wandb_run_name}_rank{accelerator.process_index}", config={
            "source_dir": args.source_dir,
            "output_h5": args.output_h5,
            "max_videos": args.max_videos,
            "resolution": args.resolution,
        })
    all_filepaths = load_file_list(args.source_dir, args.max_videos)
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    filepaths_for_this_process = all_filepaths[rank::world_size]
    sys.path.insert(0, '/dss/dsshome1/02/di38taq/droll/ldm')
    from ldm.util import instantiate_from_config
    config = OmegaConf.load(args.config_path)
    pl_sd = torch.load(args.ckpt_path, map_location=device)
    sd = pl_sd["state_dict"]
    tokenizer = instantiate_from_config(config.model)
    tokenizer.load_state_dict(sd, strict=False)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    tokenizer = tokenizer.to(device)
    output_h5 = f"{os.path.splitext(args.output_h5)[0]}_rank{rank}.h5"
    process_videos(filepaths_for_this_process, device, tokenizer, args, output_h5)
    accelerator.wait_for_everyone()
    if rank == 0:
        print("Processing complete. Merge H5 files if needed.")
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
