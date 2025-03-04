import json
from pathlib import Path
import os, sys
from omegaconf import OmegaConf
import torch
import time
from itertools import islice
from collections import defaultdict


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from celebv.dali_video_dataset import DALIVideoDataset, shuffled
except:
    from datasets.celebv.dali_video_dataset import DALIVideoDataset, shuffled


class CelebVDataset(DALIVideoDataset):
    def __init__(
        self,
        data_path,
        split,
        video_frames=16,  # snippet_duration=16,  # video_frames
        fps=25,
        batch_size=8,
        ##########################################
        dataset_type="wds",
        units="frames",
        snippet_shift=False,
        n_snippets_per_video=None,
        n_content_frames=0,
        trim=0,
        # extensions="mp4;avi",
        extensions=["mp4;avi", "svae"],
        n_threads=8,
        partial=True,
        shuffle=False,
        n_epochs=1,
        device="cpu",
        device_id=0,
        shard_id=0,
        n_shards=1,
        video_shuffle_size=1024,
        snippet_shuffle_size=1024,
        initial_snippet_shuffle_size=None,
        last_batch_policy="cut",
        deterministic=True,
        decoding_batch_size=None,
        decoding_threads=None,
        decoding_device=None,
        decoding_device_id=None,
        global_seed=1337,
        **kwargs,
    ):

        video_metadata_path = os.path.join(data_path, "videos.json")
        assert split in ["train", "val"], f"Split must be 'train' or 'val', got {split}"
        tars = [str(tar) for tar in Path(data_path).rglob("*.tar")]
        assert len(tars) > 0, f"No tar files found in {data_path}"
        with open(video_metadata_path, "r") as fp:
            videos_metadata = json.load(fp)

        super().__init__(
            tars,
            videos_metadata=videos_metadata,
            dataset_type=dataset_type,
            snippet_duration=video_frames,
            units=units,
            snippet_shift=snippet_shift,
            n_snippets_per_video=n_snippets_per_video,
            n_content_frames=n_content_frames,
            fps=fps,
            trim=trim,
            extensions=extensions,
            batch_size=batch_size,
            n_threads=n_threads,
            partial=partial,
            shuffle=shuffle,
            n_epochs=n_epochs,
            device=device,
            device_id=device_id,
            shard_id=shard_id,
            n_shards=n_shards,
            video_shuffle_size=video_shuffle_size,
            snippet_shuffle_size=snippet_shuffle_size,
            initial_snippet_shuffle_size=initial_snippet_shuffle_size,
            last_batch_policy=last_batch_policy,
            deterministic=deterministic,
            decoding_batch_size=decoding_batch_size,
            decoding_threads=decoding_threads,
            decoding_device=decoding_device,
            decoding_device_id=decoding_device_id,
            global_seed=global_seed,
        )

    def snippet_iterator(self):
        """
        An iterator that processes decoded videos and splits them up into samples of a specified length (in seconds). If snippet_shift is true, those samples are shifted randomly and may overlap.
        Returns split up samples along with frame timestamps and the video name.
        """
        device = (
            torch.device("cuda", self.device_id)
            if self.device == "gpu"
            else torch.device("cpu")
        )

        source = self.video_iterator(self._local_files, shuffle=True)

        source = self.make_snippets(source)

        if self.shuffle and self.snippet_shuffle_size > 1:
            # TODO: if decoding one gpu, then shuffling might cause OOM really quickly.
            # TODO: transfer to cpu before shuffling (advanced: transfer to pinned memory non-blocking and synchreonize cuda stream when picking sample from shuffle buffer)
            source = shuffled(
                source,
                self.snippet_shuffle_size,
                initial=self.initial_snippet_shuffle_size,
                rng=self.local_rng,
            )

        # TODO: GPU --> CPU non_blocking is dangerous as the cuda stream must additionally be synchronized
        non_blocking = self.device == "gpu"

        snippets_dict, timesteps, names, paths = defaultdict(list), [], [], []

        if self.n_content_frames > 0:
            content_frames_b = defaultdict(list)

        for data in source:
            if self.n_content_frames > 0:
                snippet_dict, timestep, content_frames, name, path = data
                for e, d in content_frames.items():
                    content_frames_b[e].append(d.to(device, non_blocking=non_blocking))
            else:
                snippet_dict, timestep, name, path = data

            for e, d in snippet_dict.items():
                snippets_dict[e].append(d.to(device, non_blocking=non_blocking))

            timesteps.append(timestep.to(device, non_blocking=non_blocking))
            names.append(name)
            paths.append(path)

            if len(timesteps) >= self.batch_size:
                if self.n_content_frames > 0:
                    d = {k: torch.stack(v) for k, v in snippets_dict.items()}
                    d["timesteps"] = torch.stack(timesteps)  # dongzhuoyao
                    d["content_frames"] = {
                        k: torch.stack(v) for k, v in content_frames_b.items()
                    }
                    d["names"] = names
                    d["paths"] = paths
                    yield d
                    snippets_dict, timesteps, content_frames_b, names, paths = (
                        defaultdict(list),
                        [],
                        defaultdict(list),
                        [],
                        [],
                    )
                else:
                    d = {k: torch.stack(v) for k, v in snippets_dict.items()}
                    d["timesteps"] = torch.stack(timesteps)
                    d["names"] = names
                    d["paths"] = paths
                    yield d
                    snippets_dict, timesteps, names, paths = (
                        defaultdict(list),
                        [],
                        [],
                        [],
                    )

        if (len(timesteps) == self.batch_size) or (len(timesteps) > 0 and self.partial):
            if self.n_content_frames > 0:
                d = {k: torch.stack(v) for k, v in snippets_dict.items()}
                d["timesteps"] = torch.stack(timesteps)
                d["content_frames"] = {
                    k: torch.stack(v) for k, v in content_frames_b.items()
                }
                d["names"] = names
                d["paths"] = paths
                yield d
            else:
                d = {k: torch.stack(v) for k, v in snippets_dict.items()}
                d["timesteps"] = torch.stack(timesteps)
                d["names"] = names
                d["paths"] = paths
                yield d


if __name__ == "__main__":
    if True:
        data_path = "/export/compvis-nfs/group/datasets/CelebVHQ/wds-512px-15s-nobb"
    elif False:
        data_path = (
            "/export/compvis-nfs/group/datasets/CelebVHQ/wds-256px-15sec-vids_svae"
        )

    if True:
        dataset = CelebVDataset(
            data_path=data_path,
            split="train",
            extensions=["mp4;avi", "svae"],
            batch_size=8,
        )
        # shape,[B, T, C, H, W]
        for data in dataset:
            for k, v in data.items():
                try:
                    print(f"{k:10} {v.shape} {v.min():.2f} {v.max():.2f} {v.std():.2f}")
                except:
                    print(f"{k:10} {len(v)}")
            break
    elif False:
        config = OmegaConf.load("config/data/celebv256.yaml")
        dataset = CelebVDataset(
            # data_path=data_path,
            # split="val",
            # extensions=["mp4;avi", "svae"],
            # batch_size=8,
            **config,
        )
        # shape,[B, T, C, H, W]
        for data in dataset:
            for k, v in data.items():
                try:
                    print(f"{k:10} {v.shape} {v.min():.2f} {v.max():.2f} {v.std():.2f}")
                except:
                    print(f"{k:10} {len(v)}")
            break
    elif True:
        from omegaconf import OmegaConf

        config = OmegaConf.load("config/data/celebv256.yaml")
        datamod = WebDataModuleFromConfig(**config)
        dataloader = datamod.train_dataloader()

        def show_key_details(batch, key):
            print(key, batch[key].shape, batch[key].max(), batch[key].min())

        for i, batch in enumerate(dataloader):
            print(batch.keys())
            print(
                batch["frame_feature256"].shape,
                batch["frame_feature256"].max(),
                batch["frame_feature256"].min(),
            )  # [B,T,C,H,W]
            # show_key_details(batch, "cls_id")
            # show_key_details(batch, "cls_name")
            # print(batch["emotions_caption"])
            break
        print("end")
