import io
import os
import sys
import math
import random
import time
from collections import defaultdict
from itertools import islice
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIRaggedIterator
import pathlib
import numpy as np
import torch
from torch.nn.functional import interpolate
import torch.distributed

try:
    import datasets.celebv.utils as utils
except:
    import celebv.utils as utils


@pipeline_def
def wds_video_pipe(
    paths,
    ext,
    device,
    index_files=None,
    shuffle=False,
    shuffle_size=1024,
):
    raws = fn.readers.webdataset(
        paths=paths,
        ext=ext,
        index_paths=index_files,
        random_shuffle=shuffle,
        initial_fill=shuffle_size,
        missing_component_behavior="skip",
        name="wds",
        pad_last_batch=False,
    )
    if len(ext) == 1:
        source_info = fn.get_property(raws, key="source_info")
        video = fn.experimental.decoders.video(raws, device=device, name="decoder")
        return video, source_info
    else:
        source_info = fn.get_property(raws[0], key="source_info")
        video = fn.experimental.decoders.video(raws[0], device=device, name="decoder")
        return video, *raws[1:], source_info


@pipeline_def()
def wds_read_pipe(
    paths,
    ext,
    index_files=None,
    shuffle=False,
    shuffle_size=1024,
):
    outs = fn.readers.webdataset(
        paths=paths,
        ext=ext,
        index_paths=index_files,
        random_shuffle=shuffle,
        initial_fill=shuffle_size,
        missing_component_behavior="skip",
        name="wds",
        pad_last_batch=False,
    )
    if len(ext) == 1:
        source_info = fn.get_property(outs, key="source_info")
        return outs, source_info
    else:
        source_info = fn.get_property(outs[0], key="source_info")
        return *outs, source_info


# @pipeline_def
# def video_decoder_pipe(source, num_outputs, device):
#     inp = fn.external_source(source, num_outputs=num_outputs)
#     video = fn.experimental.decoders.video(inp[0], device=device, name='decoder')
#     return video, *inp[1:]


@pipeline_def
def default_preprocess_pipe(
    source,
    size,
    device,
    resized_affine_crop=False,
):
    if resized_affine_crop:
        video, timesteps, video_name, affine_mat = fn.external_source(
            source, num_outputs=4, device=device
        )
        video = fn.warp_affine(video, affine_mat, size=size, device=device)
    else:
        video, timesteps, video_name = fn.external_source(
            source, num_outputs=3, device=device
        )
        video = fn.resize(
            video,
            resize_x=size,
            resize_y=size,
            antialias=True,
            device=device,
        )
    return video, timesteps, video_name


class DALIVideoDataset:
    """
    Video dataset built on NVIDIA's DALI framework for optimal performance.

    To modify the dataset for a specific use-case, it probably suffices to inherit from this class and to override the ``preprocess`` function.

    Supports sharding for multi-GPU training to equally distribute the tar files among the nodes and ensures that all nodes get exactly the same amount of batches per epoch.

    Supports data shuffling at the video snippet level (see `sample_iterator`).
    """

    def __init__(
        self,
        files,
        videos_metadata,
        dataset_type="wds",
        snippet_duration=None,
        units="seconds",
        snippet_shift=False,
        n_snippets_per_video=None,
        n_content_frames=0,
        fps=None,
        trim=0,
        extensions="mp4;avi",
        batch_size=8,
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
    ):
        super().__init__()

        assert dataset_type in ["wds"]
        self.dataset_type = dataset_type

        # if isinstance(extensions, str):
        extensions = list(extensions)

        if isinstance(files, (tuple, list)):
            self.files = files

        self.videos_metadata = videos_metadata

        self.snippet_dur = (
            snippet_duration or snippet_duration if snippet_duration > 0 else None
        )
        assert units in ["seconds", "frames"]
        self.units = units
        self.snippet_shift = snippet_shift
        assert n_snippets_per_video is None or n_snippets_per_video > 0
        self.n_snippets_per_video = n_snippets_per_video
        self.n_content_frames = n_content_frames
        self.fps = fps
        self.trim = trim

        self.ext = extensions

        self.batch_size = batch_size
        self.n_threads = n_threads
        self.partial = partial
        self.n_epochs = n_epochs
        self.device = device
        self.device_id = device_id
        self.n_shards = n_shards
        self.shard_id = shard_id
        self.shuffle = shuffle
        self.video_shuffle_size = video_shuffle_size
        self.snippet_shuffle_size = snippet_shuffle_size
        self.initial_snippet_shuffle_size = initial_snippet_shuffle_size or 100
        self.deterministic = deterministic
        self.last_batch_policy = last_batch_policy

        self.decoding_device = decoding_device or self.device
        self.decoding_batch_size = decoding_batch_size or self.batch_size
        self.decoding_threads = decoding_threads or self.n_threads
        self.decoding_device_id = decoding_device_id or self.device_id

        self.epoch = 0
        assert global_seed is not None
        self.seed = global_seed

        self.initialize()

    def initialize(self):
        """
        Compute information necessary for distributing an equal amount of batches among all nodes.
        """
        n_remove = 0
        n_keep = 0
        n_total_frames = 0
        total_dur = 0
        n_samples_per_file = defaultdict(list)

        if self.dataset_type == "wds":
            index_files = []
            for tar in self.files:
                idx_file = pathlib.Path(tar).with_suffix(".idx")
                if idx_file.exists():
                    index_files.append(str(idx_file))

            index_files = index_files if len(index_files) == len(self.files) else None

            if index_files is not None and len(index_files) == len(self.files):
                from pathlib import Path

                def info_iterator():
                    for index_file in index_files:
                        with open(index_file, "r") as fp:
                            names = [
                                ".".join(l.strip().split(" ")[3].split(".")[:-1])
                                for l in fp.readlines()[1:]
                            ]
                        tar_path = str(Path(index_file).with_suffix(".tar"))
                        for name in names:
                            yield name, str(tar_path)

            else:

                def info_iterator():
                    it = self.wds_iterator(self.files, False)
                    for *_, video_names, paths in it:
                        for video_name, path in zip(video_names, paths):
                            yield video_name, path

            for video_name, path in info_iterator():
                if self.filter(video_name):
                    duration_secs = self.videos_metadata[video_name]["duration"]
                    fps = self.fps or self.videos_metadata[video_name]["fps"]

                    if self.units == "frames":
                        dur = int(duration_secs * fps)
                    else:
                        dur = duration_secs
                    dur -= 2 * self.trim

                    n_samples_per_file[path].append(self.get_num_snippets(dur))
                    n_keep += 1
                    n_total_frames += int(duration_secs * fps)
                    total_dur += self.videos_metadata[video_name]["duration"]
                else:
                    n_remove += 1

        self.files = list(n_samples_per_file.keys())

        self.n_samples_per_file = {
            file: sum(samples) for file, samples in n_samples_per_file.items()
        }

        self.n_total_frames = n_total_frames

        print(f"[{self.shard_id} / {self.n_shards}] Initialized dataset")
        print(f"[{self.shard_id}] Ignoring {n_remove} / {n_remove + n_keep} videos")
        print(
            f"[{self.shard_id}] Total samples: {sum(self.n_samples_per_file.values())}"
        )
        print(f"[{self.shard_id}] Total duration: {total_dur / 3600:.1f} hours")
        # print(f"Total frames: {n_total_frames}")

    def filter(self, video_name):
        if (
            video_name not in self.videos_metadata
            or "duration" not in self.videos_metadata[video_name]
            or "fps" not in self.videos_metadata[video_name]
        ):
            return False

        if self.snippet_dur is None:
            return True

        if self.units == "frames":
            scale = self.videos_metadata[video_name]["fps"]
        else:
            scale = 1

        return (
            (self.videos_metadata[video_name]["duration"] * scale) - (2 * self.trim)
        ) >= self.snippet_dur

    def initialize_next_epoch(self):
        self.epoch = self.epoch + 1

        print(f"[{self.shard_id}] Initiating epoch {self.epoch}")

        # Inititialize global RNG for operations that must be the same on all nodes.
        # NOTE: Use with care and ensure all nodes use it exactly the same amount of times.
        # Currently only used for shard shuffling, thus only local variable for safety.
        if self.deterministic:
            self.global_seed = utils.make_seed(self.seed, self.epoch)
        else:
            self.global_seed = utils.make_seed(
                self.seed,
                self.epoch,
                os.getpid(),
                time.time_ns(),
                os.urandom(4),
            )
        global_rng = random.Random(self.global_seed)

        # Initialize RNG for operations that may differ among nodes.
        if self.deterministic:
            self.local_seed = utils.make_seed(self.seed, self.epoch, self.shard_id)
        else:
            self.local_seed = utils.make_seed(
                self.seed,
                self.epoch,
                self.shard_id,
                os.getpid(),
                time.time_ns(),
                os.urandom(4),
            )
        self.local_rng = random.Random(self.local_seed)

        # Split up tar files into individual shards. Each shard gets used by exactly one node, respectively.
        if self.shuffle:
            global_rng.shuffle(self.files)

        files_per_shard = {
            shard_id: list(islice(self.files, shard_id, None, self.n_shards))
            for shard_id in range(self.n_shards)
        }

        batches_per_shard = {
            shard_id: sum(
                self.n_samples_per_file[tar_file]
                for tar_file in files_per_shard[shard_id]
            )
            / self.batch_size
            for shard_id in range(self.n_shards)
        }

        if self.partial:
            batches_per_shard = {
                k: int(math.ceil(v)) for k, v in batches_per_shard.items()
            }
        else:
            batches_per_shard = {
                k: int(math.floor(v)) for k, v in batches_per_shard.items()
            }

        # To ensure that all nodes see the same amount of batches per epoch, we use the size of the smallest shard for *all* nodes.
        self._n_min_batches = min(batches_per_shard.values())
        self._n_max_batches = max(batches_per_shard.values())
        self._n_batches = batches_per_shard[self.shard_id]
        self._local_files = files_per_shard[self.shard_id]
        print(f"[{self.shard_id}] num_batches: {self._n_batches}")

    def __iter__(self):
        for _ in range(self.n_epochs):
            self.initialize_next_epoch()
            it = self.preprocess(self.snippet_iterator())
            if self.last_batch_policy == "cut":
                yield from islice(it, self._n_min_batches)
            elif self.last_batch_policy == "pad":
                yield from islice(it, self._n_batches)
                for _ in range(self._n_max_batches - self._n_batches):
                    yield None

    def preprocess(self, source):
        yield from source

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
                    yield dict(snippets_dict), timesteps, dict(
                        content_frames_b
                    ), names, paths
                    snippets_dict, timesteps, content_frames_b, names, paths = (
                        defaultdict(list),
                        [],
                        defaultdict(list),
                        [],
                        [],
                    )
                else:
                    yield dict(snippets_dict), timesteps, names, paths
                    snippets_dict, timesteps, names, paths = (
                        defaultdict(list),
                        [],
                        [],
                        [],
                    )

        if (len(timesteps) == self.batch_size) or (len(timesteps) > 0 and self.partial):
            if self.n_content_frames > 0:
                yield dict(snippets_dict), timesteps, dict(
                    content_frames_b
                ), names, paths
            else:
                yield dict(snippets_dict), timesteps, names, paths

    def make_snippets(self, source):
        vid_ext = self.ext[0]
        half_interval = self.snippet_dur / 2

        for data_dict, names, paths in source:
            for i, (name, path) in enumerate(zip(names, paths)):
                # All video representations must have the same number of frames
                assert all(
                    data_dict[e][i].shape[0] == data_dict[vid_ext][i].shape[0]
                    for e in self.ext
                ), f"{name}: Extensions have varying amounts of frames"

                n_frames = data_dict[vid_ext][i].shape[0]
                fps = float(self.videos_metadata[name]["fps"])

                # Optionally resample video
                if self.fps is not None:
                    # T, H, W, C --> H*W, C, T
                    for e_idx, e in enumerate(self.ext):
                        if e_idx == 0:
                            ts, hs, ws, cs = data_dict[e][i].shape
                            d = (
                                data_dict[e][i]
                                .to(torch.float32)
                                .permute(1, 2, 3, 0)
                                .flatten(0, 1)
                            )
                            d = interpolate(
                                d, scale_factor=float(self.fps / fps), mode="nearest"
                            )
                            d = d.reshape(hs, ws, cs, -1).permute(3, 0, 1, 2)
                        else:
                            ts, cs, hs, ws = data_dict[e][i].shape
                            d = (
                                data_dict[e][i]
                                .to(torch.float32)
                                .permute(2, 3, 1, 0)
                                .flatten(0, 1)
                            )
                            d = interpolate(
                                d, scale_factor=float(self.fps / fps), mode="nearest"
                            )
                            d = d.reshape(hs, ws, cs, -1).permute(3, 2, 0, 1)
                        data_dict[e][i] = d

                    new_n_frames = data_dict[self.ext[0]][i].shape[0]
                    fps = fps * (new_n_frames / n_frames)
                    n_frames = new_n_frames

                if self.units == "seconds":
                    duration = n_frames / fps
                elif self.units == "frames":
                    duration = n_frames

                timesteps = torch.arange(
                    n_frames, device=data_dict[vid_ext][0].device, dtype=torch.float32
                ).div_(fps)

                # Optionally trim start and end of video
                if self.trim > 0:
                    if self.units == "seconds":
                        start_idx = (
                            torch.searchsorted(
                                timesteps, torch.tensor(self.trim), side="left"
                            )
                            + 1
                        )
                        end_idx = torch.searchsorted(
                            timesteps, torch.tensor(duration - self.trim), side="right"
                        )
                    else:
                        start_idx = self.trim
                        end_idx = n_frames - self.trim
                    for e in self.ext:
                        data_dict[e][i] = data_dict[e][i][start_idx:end_idx]
                    timesteps = timesteps[start_idx:end_idx]
                    if self.units == "seconds":
                        duration = duration * (timesteps.shape[0] / n_frames)
                    elif self.units == "frames":
                        duration = timesteps.shape[0]
                    n_frames = timesteps.shape[0]

                # Return entire video ...
                if self.snippet_dur is None:
                    snippet_dict = {e: data_dict[e][i].contiguous() for e in self.ext}
                    snippet_dict = {e: v.to("cpu") for e, v in snippet_dict.items()}
                    timesteps = timesteps.to("cpu")
                    # TODO: support for content frames here too
                    yield snippet_dict, timesteps, name, path

                # ... or snippets
                else:
                    for sample_idx in range(self.get_num_snippets(duration)):
                        if self.n_snippets_per_video is None:
                            shift = (
                                (
                                    (self.local_rng.random() * self.snippet_dur)
                                    - half_interval
                                )
                                if self.snippet_shift
                                else 0
                            )
                            if self.units == "frames":
                                shift = int(shift)
                            offset = sample_idx * self.snippet_dur + shift
                        else:
                            offset = self.local_rng.random() * (
                                duration - self.snippet_dur
                            )

                        if offset > (duration - self.snippet_dur):
                            offset = max(0, duration - self.snippet_dur)
                        elif offset < 0:
                            offset = min(
                                duration - self.snippet_dur, offset + half_interval
                            )
                        assert (
                            0 <= offset <= duration - self.snippet_dur
                        ), f"Offset {offset} not in [{0}, {duration - self.snippet_dur}]. End: {duration}, SL: {self.snippet_dur}"

                        if self.units == "seconds":
                            start = min(round(offset * fps), n_frames)
                            end = min(
                                round((offset + self.snippet_dur) * fps), n_frames
                            )
                        else:
                            start = min(offset, n_frames)
                            end = min(offset + self.snippet_dur, n_frames)

                        start, end = int(start), int(end)
                        snippet_timesteps = timesteps[start:end] - timesteps[start]

                        snippet_dict = {
                            e: data_dict[e][i][start:end].contiguous() for e in self.ext
                        }

                        if self.n_content_frames > 0:
                            content_idcs = torch.tensor(
                                [
                                    self.local_rng.randint(0, n_frames - 1)
                                    for _ in range(self.n_content_frames)
                                ],
                                device=snippet_timesteps.device,
                            )
                            content_frames = {
                                e: data_dict[e][i][content_idcs] for e in self.ext
                            }

                        if self.snippet_shuffle_size > 0 and self.shuffle:
                            snippet_dict = {
                                e: v.to("cpu") for e, v in snippet_dict.items()
                            }
                            if self.n_content_frames > 0:
                                content_frames = {
                                    e: v.to("cpu") for e, v in content_frames.items()
                                }
                            snippet_timesteps = snippet_timesteps.to("cpu")

                        if self.n_content_frames > 0:
                            yield snippet_dict, snippet_timesteps, content_frames, name, path
                        else:
                            yield snippet_dict, snippet_timesteps, name, path

    def video_iterator(self, files, shuffle):
        """
        An iterator that yields decoded videos among other requested extensions within the webdataset.
        """
        # device = (
        #     torch.device('cuda', self.device_id)
        #     if self.device == 'gpu' else
        #     torch.device('cpu')
        # )

        index_files = []
        for tar in files:
            idx_file = pathlib.Path(tar).with_suffix(".idx")
            if idx_file.exists():
                index_files.append(str(idx_file))

        index_files = index_files if len(index_files) == len(files) else None

        pipe = wds_video_pipe(
            files,
            self.ext,
            index_files=index_files,
            shuffle=shuffle and self.shuffle and self.video_shuffle_size > 1,
            shuffle_size=self.video_shuffle_size,
            num_threads=self.decoding_threads,
            batch_size=self.decoding_batch_size,
            device="mixed" if self.decoding_device == "gpu" else "cpu",
            device_id=self.decoding_device_id,
            seed=self.local_seed if shuffle else 0,
        )
        pipe.build()
        outputs = [e for e in self.ext] + ["source_info"]
        output_types = [
            DALIRaggedIterator.SPARSE_LIST_TAG for _ in range(len(self.ext) + 1)
        ]
        it = DALIRaggedIterator(
            pipe,
            output_map=outputs,
            output_types=output_types,
            auto_reset=False,
            reader_name="wds",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

        out_buffers, name_buffer, tar_buffer = defaultdict(list), [], []

        for data in it:
            source_infos = data[0]["source_info"]
            for i in range(len(source_infos)):
                tar_file, _, vid_name = (
                    np.array(source_infos[i]).tobytes().decode().split(":")
                )
                vid_name, _ = os.path.splitext(vid_name)

                if not self.filter(vid_name):
                    continue

                out_buffers[self.ext[0]].append(data[0][self.ext[0]][i])
                for e in self.ext[1:]:
                    out_buffers[e].append(
                        torch.from_numpy(
                            np.load(io.BytesIO(data[0][e][i].numpy().tobytes()))
                        ).to(device=data[0][self.ext[0]][i].device, non_blocking=True)
                    )

                tar_buffer.append(tar_file)
                name_buffer.append(vid_name)

                if len(name_buffer) >= self.decoding_batch_size:
                    yield dict(out_buffers), name_buffer, tar_buffer
                    out_buffers, name_buffer, tar_buffer = defaultdict(list), [], []

        if len(name_buffer) > 0:
            yield dict(out_buffers), name_buffer, tar_buffer

    def wds_iterator(self, files, shuffle=False, encode_strings=False):
        index_files = []
        for tar in files:
            idx_file = pathlib.Path(tar).with_suffix(".idx")
            if idx_file.exists():
                index_files.append(str(idx_file))

        index_files = index_files if len(index_files) == len(files) else None

        pipe = wds_read_pipe(
            files,
            self.ext,
            index_files=index_files,
            shuffle=shuffle and self.shuffle and self.video_shuffle_size > 1,
            shuffle_size=self.video_shuffle_size,
            num_threads=self.decoding_threads,
            batch_size=self.decoding_batch_size,
            device_id=self.decoding_device_id,
            seed=self.local_seed if shuffle else 0,
        )
        pipe.build()
        outputs = [e for e in self.ext] + ["source_info"]
        output_types = [
            DALIRaggedIterator.SPARSE_LIST_TAG for _ in range(len(self.ext) + 1)
        ]
        it = DALIRaggedIterator(
            pipe,
            output_map=outputs,
            output_types=output_types,
            auto_reset=False,
            reader_name="wds",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

        out_buffers, name_buffer, tar_buffer = defaultdict(list), [], []

        for data in it:
            source_infos = data[0]["source_info"]
            for i in range(len(source_infos)):
                tar_file, _, vid_name = (
                    np.array(source_infos[i]).tobytes().decode().split(":")
                )
                vid_name, _ = os.path.splitext(vid_name)

                if not self.filter(vid_name):
                    continue

                for e in self.ext:
                    out_buffers[e].append(data[0][e][i])

                if encode_strings:
                    vid_name = np.frombuffer(bytes(vid_name, "UTF-8"), dtype=np.uint8)
                    tar_file = np.frombuffer(bytes(tar_file, "UTF-8"), dtype=np.uint8)

                tar_buffer.append(tar_file)
                name_buffer.append(vid_name)

                if len(name_buffer) >= self.decoding_batch_size:
                    yield out_buffers, name_buffer, tar_buffer
                    out_buffers, name_buffer, tar_buffer = defaultdict(list), [], []

        if len(name_buffer) > 0:
            yield out_buffers, name_buffer, tar_buffer

    def get_num_snippets(self, duration):
        if self.snippet_dur is None:
            return 1
        elif self.n_snippets_per_video is not None:
            return self.n_snippets_per_video
        if duration < self.snippet_dur:
            return 0
        return int(round(duration / self.snippet_dur))


def pick(buf: list, rng: random.Random):
    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()
    return sample


def shuffled(data, bufsize=1000, initial=100, rng=None, handler=None):
    if rng is None:
        rng = random.Random(int((os.getpid() + time.time()) * 1e9))
    initial = min(initial, bufsize)
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) < bufsize:
            try:
                buf.append(next(data))  # skipcq: PYL-R1708
            except StopIteration:
                pass
        if len(buf) >= initial:
            yield pick(buf, rng)
    while len(buf) > 0:
        yield pick(buf, rng)


import re
import braceexpand


def envlookup(m):
    key = m.group(1)
    key = "WDS_" + key
    assert key in os.environ, f"missing environment variable wds_{key}"
    return os.environ[key]


def envsubst(s):
    return re.sub(r"\$\{(\w+)\}", envlookup, s)


def expand_urls(urls):
    if isinstance(urls, str):
        urllist = urls.split("::")
        result = []
        for url in urllist:
            for i in range(10):
                last = url
                url = envsubst(url)
                if url == last:
                    break
            result.extend(braceexpand.braceexpand(url))
        return result
    else:
        return list(urls)


if __name__ == "__main__":
    import json
    from pathlib import Path

    # data_dir = '/export/compvis-nfs/group/datasets/CelebVHQ/wds-256px-15.00s-nobb/val'
    data_dir = Path(
        "/export/compvis-nfs/group/datasets/CelebVHQ/wds-512px-15s-nobb/val"
    )
    tars = [str(tar) for tar in data_dir.rglob("*.tar")]

    with open(
        "/export/compvis-nfs/group/datasets/CelebVHQ/wds-512px-15s-nobb/videos.json",
        "r",
    ) as fp:
        videos_metadata = json.load(fp)

    extensions = ["mp4;avi", "svae"]

    dataset = DALIVideoDataset(
        tars,
        videos_metadata=videos_metadata,
        dataset_type="wds",
        snippet_duration=16,  # we sample 16 frame long snippets
        snippet_shift=True,  # snippets sampled from a video are randomly shifted (so they might overlap slightly)
        units="frames",  # the 'snippet_duration' and 'trim' params specify frames, not seconds
        n_content_frames=0,  # additionally sample two frames randomly from a video, TODO: support when snippet_dur is None
        fps=25,  # resample all videos to 25 fps, can be None to keep original fps
        trim=0,  # we do not trim anything from the start and end of videos
        shuffle=False,  # no shuffling
        partial=True,  # we keep the last batch
        shard_id=0,
        n_shards=1,  # load balancing: specify multiple partitions for each gpu
        n_epochs=1,
        extensions=extensions,
        batch_size=16,
        n_threads=8,
        device="gpu",  # preprocessing and output device
        device_id=1,
        decoding_device="cpu",  # NOTE: if gpu, will be transferred to cpu for shuffling and then back
        decoding_batch_size=8,
        decoding_threads=8,
    )

    n_samples = 0
    start = time.time()
    for i, (data_dict, timesteps, content_frames, names, paths) in islice(
        enumerate(iter(dataset)), 100
    ):
        videos = data_dict[extensions[0]]
        assert all(v.shape[0] == 16 for v in videos)
        n_samples += len(videos)
        print(i)
    dur = 1000 * (time.time() - start)
    dur_per_sample = dur / n_samples
    print(f"{dur_per_sample:.2f} ms per sample")
