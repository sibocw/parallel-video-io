import torch
import logging
import re
import imageio.v2 as imageio
import numpy as np
from multiprocessing import cpu_count
from time import time
from typing import Callable
from torchcodec.decoders import VideoDecoder
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from .video_io import get_video_metadata


class VideoCollectionDataset(IterableDataset):
    def __init__(
        self,
        paths: list[Path | str],
        *,
        as_image_dirs: bool = False,
        frame_sorting: None | str = None,
        transform: Callable | None = None,
        buffer_size: int = 64,
        cache_video_metadata: bool = True,
        use_cached_video_metadata: bool = True,
        n_frame_counting_workers: int = -1,
        min_frames_per_worker: int = 300,
        logger: logging.Logger | None = None,
    ):
        r"""Yields individual frames from several videos. Each "video" can be either a
        video file or a directory containing individual frames as images.

        Args:
            paths (list[Path | str]): List of video paths, or directories containing
                frames as individual images.
            as_image_dirs (bool): If True, treat each path as a directory containing
                individual frames. Otherwise, treat it as a video file.
            frame_sorting (str | None): When `as_image_dirs` is True, this argument
                specifies how images within each directory should be sorted. If None,
                files are sorted by name. If given as a string, it is used as a regex
                pattern to extract frame numbers from filenames
                (e.g. r"frame\D*(\d+)(?!\d)"). Ignored when `as_image_dirs` is False.
            transform (Callable | None): A function that is to be applied to each frame
                after loading. Note that the following operations are already applied to
                each frame:
                    (i) conversion from numpy array to torch tensor,
                    (ii) conversion from HWC to CHW format, and
                    (iii) conversion from uint8 in [0, 255] to float in [0, 1].
                The transform function, if provided, is applied after these operations.
            buffer_size (int): Number of frames to buffer when reading from video files.
                Buffering is not used when reading from image directories. Larger buffer
                sizes may improve performance at the cost of higher memory usage.
            cache_video_metadata (bool): Whether to cache video metadata to disk for
                faster subsequent loading.
            use_cached_video_metadata (bool): Whether to use cached metadata if
                available. Set to False to force re-reading metadata.
            n_frame_counting_workers (int): Number of workers to use for counting frames
                in video files. If -1, uses all available cores. Only applies when
                `as_image_dirs` is False.
            min_frames_per_worker (int): Minimum number of frames each worker should
                process. If the calculated frames per worker is below this threshold,
                the number of workers is reduced to meet this minimum. This helps avoid
                excessive overhead from too many workers on small datasets.
            logger (logging.Logger | None): Logger to use for logging. If None, takes
                from `__name__`.
        """
        self._logger = logger if logger is not None else logging.getLogger(__name__)

        self.video_paths = [Path(p) for p in paths]
        self.as_image_dirs = as_image_dirs
        self.frame_sorting = frame_sorting
        self.transform = transform
        self.buffer_size = buffer_size
        self.cache_video_metadata = cache_video_metadata
        self.use_cached_video_metadata = use_cached_video_metadata
        self.min_frames_per_worker = min_frames_per_worker

        # Check if the paths are all valid
        self._logger.info("Checking if provided video paths are valid")
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
        self.sorted_frame_paths: list[list[Path]] = []
        if as_image_dirs:
            self._logger.info(
                "Video paths are actually directories containing individual frames. "
                "Sorting frames for each directory."
            )
            # Iterate over the canonical Path objects (self.video_paths) so we
            # consistently store Path keys and avoid relying on caller types
            regex = re.compile(frame_sorting) if frame_sorting else None
            for path in self.video_paths:
                all_files = [f for f in path.iterdir() if f.is_file()]
                if regex is None:
                    sorting_func = lambda f: f.name
                else:
                    sorting_func = lambda f: _get_frame_idx_from_filename(f.name, regex)
                all_files.sort(key=sorting_func)
                self.sorted_frame_paths.append(all_files)

        # Figure out number of frames per video
        self.n_frames_by_video: list[int] = self._count_n_frames_by_video(
            n_frame_counting_workers
        )
        self.n_frames_total = sum(self.n_frames_by_video)

        # Initialize attributes for worker assignment and buffering during loading
        # worker assignment format: (video_idx, start_frameid, end_frameid)
        # This will be properly initialized in assign_workers()
        self.worker_assignments: list[list[tuple[int, int, int]]] = []
        self._frames_buffer: dict[tuple[int, int], torch.Tensor] = {}

    def _count_n_frames_by_video(self, n_frame_counting_workers: int) -> list[int]:
        """Figure out how many frames there are in each video"""
        if self.as_image_dirs:
            self._logger.info("Counting number of frames in each directory of frames.")
            n_frames_by_video = [
                len(frame_paths) for frame_paths in self.sorted_frame_paths
            ]
        else:
            # Count frames in videos. This requires partially decoding the video files
            # and it can be quite slow, so we do it in parallel and use caches.
            self._logger.info(
                "Counting number of frames in each video. This may take a while if no "
                "cached metadata is available."
            )
            pool = Parallel(n_jobs=n_frame_counting_workers)
            n_videos = len(self.video_paths)
            self._logger.info(
                f"Loading metadata from {n_videos} videos using "
                f"n_frame_counting_workers={n_frame_counting_workers} workers "
                f"(effectively {pool._effective_n_jobs()})."
            )
            start_time = time()
            metas = pool(
                delayed(get_video_metadata)(
                    path, self.cache_video_metadata, self.use_cached_video_metadata
                )
                for path in self.video_paths
            )
            walltime = time() - start_time
            self._logger.info(
                f"Loaded metadata for {n_videos} videos in {walltime:.2f} seconds."
            )
            n_frames_by_video = [meta["n_frames"] for meta in metas]

        return n_frames_by_video

    def assign_workers(self, n_loading_workers: int) -> None:
        """Assign frames to workers for balanced parallel loading.

        This method distributes all frames across the specified number of workers such
        that each worker processes approximately the same number of frames. The
        assignment ensures contiguous frame ranges within each video to minimize seeking
        overhead during decoding.

        Args:
            n_loading_workers (int): Number of workers to assign frames to. Each worker
                will be assigned approximately the same total number of frames across
                all videos.
        """
        # Make a giant array of (video_idx, frameid) pairs
        frame_specs_all = -np.ones((self.n_frames_total, 2), dtype=np.int32)
        curr_frameid_global = 0
        for vid_idx, n_frames in enumerate(self.n_frames_by_video):
            end_frameid_global = curr_frameid_global + n_frames
            frameids_local = np.arange(n_frames)
            frame_specs_all[curr_frameid_global:end_frameid_global, 0] = vid_idx
            frame_specs_all[curr_frameid_global:end_frameid_global, 1] = frameids_local
            curr_frameid_global = end_frameid_global
        assert curr_frameid_global == self.n_frames_total
        assert not np.any(frame_specs_all == -1)

        # Dynamically balance load among workers
        n_frames_per_worker = int(np.ceil(self.n_frames_total / n_loading_workers))
        self._logger.info(
            f"Assigning {self.n_frames_total} total frames from "
            f"{len(self.video_paths)} videos to {n_loading_workers} loading workers."
        )
        if n_frames_per_worker < self.min_frames_per_worker:
            n_frames_per_worker = self.min_frames_per_worker
            n_loading_workers = int(np.ceil(self.n_frames_total / n_frames_per_worker))
            self._logger.info(
                f"n_frames_per_worker is less than "
                f"self.min_frames_per_worker ({self.min_frames_per_worker}). "
                f"This will result in many workers working on not so much data, "
                f"leading to high overhead. "
                f"Increasing n_frames_per_worker to {n_frames_per_worker} and reducing "
                f"n_loading_workers to {n_loading_workers}."
            )

        # Initialize worker assignments with the final number of workers
        self.worker_assignments = [[] for _ in range(n_loading_workers)]

        for worker_id in range(n_loading_workers):
            start_idx = worker_id * n_frames_per_worker
            end_idx = min(start_idx + n_frames_per_worker, self.n_frames_total)
            frame_specs = frame_specs_all[start_idx:end_idx, :]
            # Convert to list of (video_idx, start_frameid, end_frameid)
            if frame_specs.shape[0] == 0:
                continue
            unique_video_idxs = np.unique(frame_specs[:, 0])
            for vid_idx in unique_video_idxs:
                frameids_local = frame_specs[frame_specs[:, 0] == vid_idx, 1]
                start_frameid = frameids_local[0]
                end_frameid = frameids_local[-1] + 1  # exclusive
                self.worker_assignments[worker_id].append(
                    (vid_idx, start_frameid, end_frameid)
                )
        _nframes_total_check = 0
        for worker_chunks in self.worker_assignments:
            for _, start_frameid, end_frameid in worker_chunks:
                _nframes_total_check += end_frameid - start_frameid
        assert _nframes_total_check == self.n_frames_total, "Frame count mismatch."

    def __iter__(self):
        # Get worker info for distributed loading
        worker_info = get_worker_info()
        if worker_info is None:
            # Single process
            assert (
                len(self.worker_assignments) == 1
            ), "Using a single worker but worker assignments indicate multiple workers."
            my_chunks = self.worker_assignments[0]
        else:
            # Split videos among workers
            my_chunks = self.worker_assignments[worker_info.id]

        # Each worker sequentially decodes its assigned videos
        for video_idx, start_frame_idx, end_frame_idx in my_chunks:
            if self.as_image_dirs:
                # Read individual images
                frame_files_this_video = self.sorted_frame_paths[video_idx]
                for frame_idx, frame_file in enumerate(frame_files_this_video):
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
                        "video_idx": video_idx,
                        "frame_idx": frame_idx,
                    }
            else:
                # Use torchcodec to decode videos
                decoder = VideoDecoder(
                    self.video_paths[video_idx].as_posix(),
                    seek_mode="exact",
                    dimension_order="NCHW",
                )
                for frame_idx in range(start_frame_idx, end_frame_idx):
                    frame = self._get_frame(decoder, video_idx, frame_idx)
                    frame = frame.float() / 255.0
                    if self.transform:
                        frame = self.transform(frame)
                    yield {
                        "frame": frame,
                        "video_idx": video_idx,
                        "frame_idx": frame_idx,
                    }

    def _get_frame(self, decoder: VideoDecoder, video_idx: int, frame_idx: int):
        """Get a frame at the specified index, using buffering to reduce the number of
        decoding calls. If the frame is not in the buffer, a new batch of frames is
        decoded and stored in the buffer.

        Note that this purges the previous buffer and assumes that the next frames will
        be requested soon. Therefore, this buffering strategy only makes sense for
        sequential access patterns (which is the case using our payload assignment
        strategy)."""
        key = (video_idx, frame_idx)
        if key in self._frames_buffer:
            return self._frames_buffer[key]

        n_frames_in_video = self.n_frames_by_video[video_idx]
        max_frame_idx_to_buffer = min(frame_idx + self.buffer_size, n_frames_in_video)
        frame_indices_to_buffer = list(range(frame_idx, max_frame_idx_to_buffer))
        frames = decoder.get_frames_at(frame_indices_to_buffer)  # returns a NCHW tensor
        self._frames_buffer = {
            (video_idx, idx): frames.data[i, ...]
            for i, idx in enumerate(frame_indices_to_buffer)
        }
        return self._frames_buffer[key]

    def __len__(self):
        return self.n_frames_total


class VideoCollectionDataLoader(DataLoader):
    """DataLoader for VideoCollectionDataset with automatic worker assignment.

    This DataLoader automatically assigns video frames to workers for efficient parallel
    loading. Each worker processes approximately the same number of frames, and frame
    assignments maintain contiguous ranges within videos to minimize seeking overhead.

    The output batches contain:
    - frames: Tensor of shape (batch_size, channels, height, width)
    - video_indices: List of video indices (int) for each frame
    - frame_indices: List of frame indices (int) within each video

    Note: Custom batch_sampler and collate_fn are not supported.
    """

    def __init__(
        self,
        dataset: VideoCollectionDataset,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self._logger = logger if logger is not None else logging.getLogger(__name__)

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

        num_workers = self.num_workers
        if num_workers == 0:
            num_workers = 1

        self.dataset.assign_workers(n_loading_workers=num_workers)

    @staticmethod
    def _collate(batch):
        """Receives a list of frame dicts, returns a batched dict"""
        return {
            "frames": torch.stack([item["frame"] for item in batch]),
            "video_indices": [item["video_idx"] for item in batch],
            "frame_indices": [item["frame_idx"] for item in batch],
        }


class SimpleVideoCollectionLoader(VideoCollectionDataLoader):
    def __init__(
        self,
        paths: list[Path | str],
        as_image_dirs: bool = False,
        frame_sorting: None | str = None,
        transform: Callable | None = None,
        buffer_size: int = 32,
        cache_video_metadata: bool = True,
        use_cached_video_metadata: bool = True,
        n_frame_counting_workers: int = -1,
        min_frames_per_worker: int = 300,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """Easier parallel video loading API if you don't care too much about the
        standard "Dataset + DataLoader" pattern with torch.utils.data. Use this class
        like the DataLoader.

        Arguments are those for `VideoCollectionDataset.__init__`, plus what you would
        normally pass to `torch.utils.data.DataLoader.__init__` (as keyword arguments).
        """

        if logger is None:
            logger = logging.getLogger(__name__)

        dataset = VideoCollectionDataset(
            paths,
            as_image_dirs=as_image_dirs,
            frame_sorting=frame_sorting,
            transform=transform,
            buffer_size=buffer_size,
            cache_video_metadata=cache_video_metadata,
            use_cached_video_metadata=use_cached_video_metadata,
            n_frame_counting_workers=n_frame_counting_workers,
            min_frames_per_worker=min_frames_per_worker,
            logger=logger,
        )
        num_workers = kwargs.get("num_workers", 0)  # 0 is normal DataLoader default
        kwargs["num_workers"] = _resolve_n_workers_spec(num_workers, logger)
        super().__init__(dataset, logger=logger, **kwargs)


def _get_frame_idx_from_filename(filename: str, regex_pattern: str) -> int:
    """Extract frame index from filename using the provided regex pattern."""
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


def _resolve_n_workers_spec(n_workers: int, logger: logging.Logger) -> int:
    """Resolve number of workers from user specification. If -1, use all available
    cores, if -2, use all but one core, etc. If 0, set to 1 (we don't separately
    implement doing the work in the main thread/process; a single child will be used).
    """
    n_cpu_cores = cpu_count()
    n_workers_resolved = None

    if n_workers < -n_cpu_cores:
        pass  # cannot resolve
    elif -n_cpu_cores <= n_workers < 0:
        n_workers_resolved = n_workers + n_cpu_cores + 1
        logger.info(
            f"n_workers_spec={n_workers} interpreted as {n_workers_resolved} workers "
            f"(n_cpu_cores={n_cpu_cores})."
        )
    elif n_workers == 0:
        logger.info(
            f"n_workers_spec=0 interpreted as 1 worker. Processing in main thread not "
            f"implemented; will use a single worker process/thread instead."
        )
        return 1
    elif 0 < n_workers <= n_cpu_cores:
        return n_workers
    else:
        pass  # cannot resolve

    if n_workers_resolved is None:
        raise ValueError(
            f"Invalid n_workers_spec={n_workers}. Must be between "
            f"{-n_cpu_cores} and {n_cpu_cores} (inclusive, n_cpu_cores={n_cpu_cores})."
        )
    return n_workers_resolved
