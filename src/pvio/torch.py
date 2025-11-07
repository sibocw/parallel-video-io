import torch
import logging
import re
import imageio.v2 as imageio
import numpy as np
from collections import defaultdict
from typing import Callable
from torchcodec.decoders import VideoDecoder
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from .video_io import get_video_metadata


class VideoCollectionDataset(IterableDataset):
    """Yields individual frames from Spotlight behavior recordings, either
    from videos or from image sequences."""

    def __init__(
        self,
        paths: list[Path | str],
        as_image_dirs: bool = False,
        frame_sorting: None | str = None,
        transform: Callable | None = None,
    ):
        r"""
        Args:
            paths (list[Path]): List of to video paths, or directories
                containing frames as individual images.
            as_image_dirs (bool): If True, treat each path as a directory
                containing individual frames. Otherwise, treat it as a
                video file.
            frame_sorting (str | None): When `as_image_dirs` is True, this
                argument specifies how images within each directory should
                be sorted. If None, files are sorted by name. If given as a
                string, it is used as a regex pattern to extract frame
                numbers from filenames (e.g. r"frame\D*(\d+)(?!\d)").
                When `as_image_dirs` is False, this argument is ignored.
            transform (Callable | None): A function that is to be applied
                to each frame after loading. Note that the following
                operations are already applied to each frame:
                (i) conversion from numpy array to torch tensor,
                (ii) conversion from HWC to CHW format, and
                (iii) conversion from uint8 in [0, 255] to float in [0, 1].
                The transform function, if provided, is applied after these
                operations.
        """
        self.video_paths = [Path(p) for p in paths]
        self.worker_assignments = None
        self.as_image_dirs = as_image_dirs
        self.frame_sorting = frame_sorting
        self.n_frames_lookup = None  # Populated by assign_workers()
        self.transform = transform

        # Check if the paths are all valid
        for p in self.video_paths:
            if self.as_image_dirs:
                if not p.is_dir():
                    raise ValueError(
                        f"One of the specified paths {p} is not a valid directory. "
                        "Directories containing individual frame images are expected."
                    )
            else:
                if not p.is_file():
                    raise ValueError(
                        f"One of the specified paths {p} is not a valid file. "
                        "Video files are expected."
                    )

        # Sort images if we're loading from directories of images
        self.frame_sortings = {}
        regex = re.compile(frame_sorting) if frame_sorting else None
        if as_image_dirs:
            # Iterate over the canonical Path objects (self.video_paths) so we
            # consistently store Path keys and avoid relying on caller types
            for path in self.video_paths:
                all_files = [f for f in path.iterdir() if f.is_file()]
                if regex is None:
                    sorting_func = lambda f: f.name
                else:
                    sorting_func = lambda f: self._extract_frame_number(f.name, regex)
                # Store a new sorted list (list.sort() returns None)
                self.frame_sortings[path] = sorted(all_files, key=sorting_func)

    def assign_workers(
        self,
        n_frame_loading_workers: int,
        n_metadata_indexing_workers: int = -1,
        chunk_size: int = 1000,
    ):
        # Check how many frame loading workers we're actually using (e.g. -1 actually
        # means all available cores, so we need to figure out how many that is)
        n_frame_loading_workers_effective = Parallel(
            n_jobs=n_frame_loading_workers
        )._effective_n_jobs()
        logging.info(
            f"Caller specified {n_frame_loading_workers} workers for frame loading. "
            f"This is effectively {n_frame_loading_workers_effective} workers."
        )

        # Figure out how many frames there are in each video. This allows us to split
        # the workload more evenly among workers by the number of frames.
        if self.as_image_dirs:
            self.n_frames_lookup = {
                path: len(frames) for path, frames in self.frame_sortings.items()
            }
        else:
            # Count frames in videos. This requires partially decoding the video files
            # and it can be quite slow, so we do it in parallel and use caches.
            logging.info(
                f"Loading metadata for {len(self.video_paths)} videos. "
                "This may take a while if no cached metadata is available."
            )
            metas = Parallel(n_jobs=n_metadata_indexing_workers)(
                delayed(get_video_metadata)(path)
                for path in tqdm(self.video_paths, desc="Indexing videos", disable=None)
            )
            self.n_frames_lookup = {
                path: meta["n_frames"] for path, meta in zip(self.video_paths, metas)
            }
        all_chunks: list[tuple[Path, int, int]] = []
        for path, n_frames in self.n_frames_lookup.items():
            n_chunks = int(np.ceil(n_frames / chunk_size))
            for chunk_idx in range(n_chunks):
                start_frame_idx = chunk_idx * chunk_size
                end_frame_idx = min(start_frame_idx + chunk_size, n_frames)
                all_chunks.append((path, start_frame_idx, end_frame_idx))
        self.worker_assignments = defaultdict(list)
        for chunk_idx, chunk in enumerate(all_chunks):
            worker_id = chunk_idx % n_frame_loading_workers_effective
            self.worker_assignments[worker_id].append(chunk[0])

    def __iter__(self):
        # Get worker info for distributed loading
        worker_info = get_worker_info()
        if worker_info is None:
            # Single process
            chunks_subset = []
            for path in self.video_paths:
                chunks_subset.append((path, 0, self.n_frames_lookup[path]))
        else:
            # Split videos among workers
            video_subset = self.worker_assignments[worker_info.id]

        # Each worker sequentially decodes its assigned videos
        for video_path, start_frame_idx, end_frame_idx in video_subset:
            if self.as_image_dirs:
                # Read individual images
                frame_files = self.frame_sortings[video_path]
                for frame_idx, frame_file in enumerate(frame_files):
                    if frame_idx < start_frame_idx or frame_idx >= end_frame_idx:
                        continue
                    frame = imageio.imread(frame_file)
                    frame = torch.from_numpy(frame)
                    if frame.ndim == 2:
                        frame = frame.unsqueeze(-1)  # add channel dim
                    frame = frame.permute(2, 0, 1)  # HWC to CHW
                    frame = frame.float() / 255.0  # to float in [0, 1]
                    if self.transform:
                        frame = self.transform(frame)
                    yield {
                        "frame": frame,
                        "video_path": video_path,
                        "frame_idx": frame_idx,
                    }
            else:
                # Use torchcodec to decode videos
                decoder = VideoDecoder(video_path)
                for frame_idx in range(len(decoder)):
                    frame = decoder[frame_idx]  # returns tensor in CHW
                    frame = frame.float() / 255.0  # to float in [0, 1]
                    if self.transform:
                        frame = self.transform(frame)
                    yield {
                        "frame": frame,
                        "video_path": video_path,
                        "frame_idx": frame_idx,
                    }

    def __len__(self):
        if self.n_frames_lookup is None:
            raise ValueError(
                "VideoCollectionDataset length is unknown until workers are assigned. "
                "Call `assign_workers()` before using `len()`."
            )
        return sum(self.n_frames_lookup.values())

    @staticmethod
    def _extract_frame_number(filename: str, regex_pattern: str) -> int:
        matches = re.findall(regex_pattern, filename)
        if len(matches) != 1:
            raise ValueError(
                f"{len(matches)} matches found in filename {filename} "
                f"using regex pattern {regex_pattern}. Only one match is expected."
            )
        try:
            return int(matches[0])
        except ValueError as e:
            raise ValueError(
                f"Failed to parse '{matches[0]}' as int. This substring is extracted "
                f"from filename {filename} using regex pattern {regex_pattern}."
            ) from e


class VideoCollectionDataLoader(DataLoader):
    def __init__(
        self, dataset: VideoCollectionDataset, chunk_size: int = 1000, **kwargs
    ):
        if not isinstance(dataset, VideoCollectionDataset):
            raise ValueError(
                "VideoCollectionDataLoader only works with VideoCollectionDataset."
            )
        if kwargs.get("batch_sampler") is not None:
            raise ValueError(
                "VideoCollectionDataLoader does not support custom batch samplers."
            )
        if kwargs.get("collate_fn") is not None:
            raise ValueError(
                "VideoCollectionDataLoader must use the built-in collate function."
            )

        kwargs["collate_fn"] = self._collate
        super().__init__(dataset, **kwargs)

        self.dataset.assign_workers(
            n_frame_loading_workers=self.num_workers, chunk_size=chunk_size
        )

    @staticmethod
    def _collate(batch):
        """Receives a list of frame dicts, returns a batched dict"""
        return {
            "frames": torch.stack([item["frame"] for item in batch]),
            "video_paths": [item["video_path"] for item in batch],
            "frame_indices": [item["frame_idx"] for item in batch],
        }
