import os
import random
import argparse
import glob
from tqdm import tqdm

import numpy as np
import torch
from decord import VideoReader
from common_metrics_on_video_quality import calculate_fvd


def load_videos(
    dir, has_subset=False, resolution=256, frames=17, pos="frame", num_videos=2048
):
    if not has_subset:
        videos = os.listdir(dir)
        videos = [os.path.join(dir, v) for v in videos if v.endswith("mp4")]

    else:
        videos = glob.glob(f"{dir}/*/*.avi")

    random.shuffle(videos)
    if num_videos != -1:
        videos = videos[:num_videos]

    video_data = []
    for v in tqdm(videos):
        if v.endswith(".gif"):
            new_v = v.replace("gif", "mp4")
            os.system(f"ffmpeg -i {v} {new_v} -loglevel quiet")
            v = new_v

        vr = VideoReader(v, width=resolution, height=resolution)
        sampled_frms = (
            vr.get_batch(np.arange(0, len(vr), 1, dtype=int)).asnumpy().astype(np.uint8)
        )  # T H W 3

        assert len(vr) >= frames
        if len(vr) > frames:
            if pos == "first":
                sampled_frms = sampled_frms[:frames]

            elif pos == "last":
                sampled_frms = sampled_frms[-frames:]

            elif pos == "random":
                start_idx = random.randint(0, len(vr) - frames)
                sampled_frms = sampled_frms[start_idx:start_idx + frames]
            else:
                center = len(vr) // 2
                if frames % 2 == 0:
                    frame_indices = np.array(
                        range(center - frames // 2, center + frames // 2)
                    ).astype(int)
                else:
                    frame_indices = np.array(
                        range(center - frames // 2, center + frames // 2 + 1)
                    ).astype(int)

                sampled_frms = sampled_frms[frame_indices]

        vid_frm_array = (
            torch.from_numpy(sampled_frms).float().permute(0, 3, 1, 2)
        ) / 255.0  # T, 3, H, W
        video_data.append(vid_frm_array)

    video_data = torch.stack(video_data, dim=0)
    return video_data  # torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)


def calculate_fvd_github(gen_dir, gt_dir, resolution=256, frames=17, sampling="center"):
    num_videos_gt = len([f for f in os.listdir(gt_dir) if f.endswith(".mp4")])
    num_videos_gen = len([f for f in os.listdir(gen_dir) if f.endswith(".mp4")])
    assert num_videos_gt == num_videos_gen, f"gen videos: {num_videos_gen}, gt videos: {num_videos_gt}"
    num_videos = num_videos_gt
    if False:
        gt_videos = load_videos(
            gt_dir,
            has_subset=True,
            resolution=resolution,
            frames=frames,
            pos=sampling,
            num_videos=num_videos,
        )

    else:
        gt_videos = load_videos(
            gt_dir,
            has_subset=False,
            resolution=resolution,
            frames=frames,
            pos=sampling,
            num_videos=num_videos,
        )

    gen_videos = load_videos(
        gen_dir,
        has_subset=False,
        resolution=resolution,
        frames=frames,
        pos=sampling,
        num_videos=num_videos,
    )

    # print(gen_videos.shape, gt_videos.shape)
    results = calculate_fvd(gt_videos, gen_videos, device="cuda", method="videogpt")
    print(results)
    fvd16 = results["value"][16]
    print(f"fvd16: {fvd16}")
    return dict(fvd=fvd16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="k600", choices=["k600", "ucf"])
    parser.add_argument(
        "--gen_dir", default="./data/unittest_ffs_tokenizer_titok/gt", type=str
    )
    parser.add_argument(
        "--gt_dir",
        default="./data/unittest_ffs_tokenizer_titok/reconstructed",
        type=str,
    )
    parser.add_argument("--split", type=str, choices=["train", "test"])
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument(
        "--sampling", type=str, default="center", choices=["first", "last", "center"]
    )
    parser.add_argument("--num_videos", type=int, default=2048)
    args = parser.parse_args()

    gen_dir = args.gen_dir
    gt_dir = args.gt_dir

    if args.dataset == "ucf":
        gt_videos = load_videos(
            gt_dir,
            has_subset=True,
            resolution=args.resolution,
            frames=args.frames,
            pos=args.sampling,
            num_videos=args.num_videos,
        )

    else:
        gt_videos = load_videos(
            gt_dir,
            has_subset=False,
            resolution=args.resolution,
            frames=args.frames,
            pos=args.sampling,
            num_videos=args.num_videos,
        )

    gen_videos = load_videos(
        gen_dir,
        has_subset=False,
        resolution=args.resolution,
        frames=args.frames,
        pos=args.sampling,
        num_videos=args.num_videos,
    )

    # print(gen_videos.shape, gt_videos.shape)
    results = calculate_fvd(gt_videos, gen_videos, device="cuda", method="videogpt")
    print(results)
