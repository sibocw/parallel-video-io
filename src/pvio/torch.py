import torch
import logging
import re
import numpy as np
from sys import stderr
from multiprocessing import cpu_count
from typing import Callable
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from .io import get_video_metadata
from .video import Video, EncodedVideo, ImageDirVideo


logger = logging.getLogger(__name__)


class VideoCollectionDataset(IterableDataset):
    def __init__(
        self,
        videos: list[Video],
        *,
        transform: Callable | None = None,
        use_cached_video_metadata: bool = True,
        n_frame_counting_workers: int = -1,
        progress_bar: bool | None = None,
    ):
        r"""Yields individual frames from several videos.

        Args:
            videos (list[Video]): List of Video objects (EncodedVideo, ImageDirVideo, etc.)
            transform (Callable | None): A function that is to be applied to each frame
                after loading. Note that the following operations are already applied to
                each frame:
                    (i) conversion from numpy array to torch tensor,
                    (ii) conversion from HWC to CHW format, and
                    (iii) conversion from uint8 in [0, 255] to float in [0, 1].
                The transform function, if provided, is applied after these operations.
            use_cached_video_metadata (bool): Whether to use cached metadata if
                available. Set to False to force re-reading metadata.
            n_frame_counting_workers (int): Number of workers to use for counting frames
                in video files. If -1, uses all available cores.
            progress_bar (bool | None): Whether to show progress bar during metadata loading.
                If None, shows progress bar only when stderr is a TTY.
        """
        self.videos = videos
        self.transform = transform
        self.progress_bar = stderr.isatty() if progress_bar is None else progress_bar

        # Load video metadata before calling `.setup()` on the videos
        # This prevents redundant metadata loading if multiple Video objects are linked
        # to the same actual video
        unique_video_paths = set(
            vid.path for vid in videos if isinstance(vid, EncodedVideo)
        )
        pool = Parallel(n_jobs=n_frame_counting_workers)
        n_workers_effective = pool._effective_n_jobs()
        logger.info(
            f"Loading metadata for {len(unique_video_paths)} unique videos using "
            f"{n_frame_counting_workers} workers (effectively {n_workers_effective})."
        )
        _pbar = self.progress_bar and len(unique_video_paths) > n_workers_effective
        # Run metadata loading in parallel - no need to keep the results, we just want
        # to generate the metadata cache files
        _ = pool(
            delayed(get_video_metadata)(
                path, cache_metadata=True, use_cached_metadata=use_cached_video_metadata
            )
            for path in tqdm(
                unique_video_paths,
                file=stderr,
                disable=not _pbar,
                desc="Loading metadata",
            )
        )

        # Set up all videos and build an index of number of frames per video, using
        # the caches that we've just generated
        for video in self.videos:
            if isinstance(video, EncodedVideo) and not video.use_cached_metadata:
                logger.warning(
                    "For efficiency, `use_cached_metadata` must be set to True for "
                    "all EncodedVideo objects when they are managed by "
                    "VideoCollectionDataset. Overriding to True."
                )
                video.use_cached_metadata = True
            video.setup()

        # Build index of number of frames per video
        self.n_frames_by_video = [len(video) for video in self.videos]
        self.n_frames_total = sum(self.n_frames_by_video)

        # Initialize attributes for worker assignment - to be filled in .assign_workers
        # The i-th worker's assignments are self.worker_assignments[i], which is a list
        # of (video_id, start_vir_frame_id, end_vir_frame_id) tuples.
        self.worker_assignments: list[list[tuple[int, int, int]]] = []

    def assign_workers(
        self, n_loading_workers: int, min_frames_per_worker: int = 300
    ) -> None:
        """Assign frames to workers for balanced parallel loading.

        This method distributes all frames across the specified number of workers such
        that each worker processes approximately the same number of frames. The
        assignment ensures contiguous frame ranges within each video to minimize seeking
        overhead during decoding.

        Args:
            n_loading_workers (int): Number of workers to assign frames to. Each worker
                will be assigned approximately the same total number of frames across
                all videos.
            min_frames_per_worker (int): Minimum number of frames each worker should
                process. If the calculated frames per worker is below this threshold,
                the number of workers is reduced to meet this minimum. This helps avoid
                excessive overhead from too many workers on small datasets.
        """
        # Make an array of (video_id, vir_frame_id) pairs. frame_id is virtual (i.e.
        # index 0 corresponds to physical frame_id=frame_range[0])
        frame_specs_all = -np.ones((self.n_frames_total, 2), dtype=np.int32)
        start_global_frame_id = 0
        for video_id, n_frames in enumerate(self.n_frames_by_video):
            end_global_frame_id = start_global_frame_id + n_frames
            local_vfids = np.arange(n_frames)
            frame_specs_all[start_global_frame_id:end_global_frame_id, 0] = video_id
            frame_specs_all[start_global_frame_id:end_global_frame_id, 1] = local_vfids
            start_global_frame_id = end_global_frame_id
        assert start_global_frame_id == self.n_frames_total
        assert not np.any(frame_specs_all == -1)

        # Dynamically balance load among workers
        n_frames_per_worker = int(np.ceil(self.n_frames_total / n_loading_workers))
        logger.info(
            f"Assigning {self.n_frames_total} total frames from "
            f"{len(self.videos)} videos to {n_loading_workers} loading workers."
        )
        if n_frames_per_worker < min_frames_per_worker:
            n_frames_per_worker = min_frames_per_worker
            n_loading_workers = int(np.ceil(self.n_frames_total / n_frames_per_worker))
            logger.info(
                f"`n_frames_per_worker` is less than `min_frames_per_worker` "
                f"({min_frames_per_worker}). This will result in many workers working "
                f"on not so much data, leading to high overhead. "
                f"Increasing `n_frames_per_worker` to {n_frames_per_worker} and "
                f"reducing `n_loading_workers` to {n_loading_workers}."
            )

        # Initialize worker assignments with the final number of workers
        self.worker_assignments = [[] for _ in range(n_loading_workers)]

        for worker_id in range(n_loading_workers):
            start_global_frame_id = worker_id * n_frames_per_worker
            end_global_frame_id = min(
                start_global_frame_id + n_frames_per_worker, self.n_frames_total
            )
            my_specs = frame_specs_all[start_global_frame_id:end_global_frame_id, :]
            # Convert to list of (video_id, start_vir_frame_id, end_vir_frame_id)
            if my_specs.shape[0] == 0:
                continue
            unique_video_ids = np.unique(my_specs[:, 0])
            for video_id in unique_video_ids:
                vir_frame_ids_local = my_specs[my_specs[:, 0] == video_id, 1]
                start_vir_frame_id = vir_frame_ids_local[0]
                end_vir_frame_id = vir_frame_ids_local[-1] + 1  # exclusive
                self.worker_assignments[worker_id].append(
                    (video_id, start_vir_frame_id, end_vir_frame_id)
                )
        _nframes_total_check = 0
        for worker_chunks in self.worker_assignments:
            for _, start_vir_frame_id, end_vir_frame_id in worker_chunks:
                _nframes_total_check += end_vir_frame_id - start_vir_frame_id
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
        for video_id, start_vir_frame_id, end_vir_frame_id in my_chunks:
            video = self.videos[video_id]
            for vir_frame_id in range(start_vir_frame_id, end_vir_frame_id):
                frame = video.read_frame(vir_frame_id, transform=self.transform)
                yield {"frame": frame, "video_id": video_id, "frame_id": vir_frame_id}

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
        min_frames_per_worker: int = 300,
        **kwargs,
    ):
        """Create a torch.utils.data.DataLoader compatible for VideoCollectionDataset.

        Args:
            dataset (VideoCollectionDataset): The dataset to load from.
            min_frames_per_worker (int): Minimum number of frames each worker should
                process. If the calculated frames per worker is below this threshold,
                the number of workers is reduced to meet this minimum. This helps avoid
                excessive overhead from too many workers on small datasets.
            **kwargs: Additional keyword arguments passed to the base DataLoader.
        """
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

        self.dataset.assign_workers(
            n_loading_workers=num_workers, min_frames_per_worker=min_frames_per_worker
        )

    @staticmethod
    def _collate(batch):
        """Receives a list of frame dicts, returns a batched dict"""
        return {
            "frames": torch.stack([item["frame"] for item in batch]),
            "video_indices": [item["video_id"] for item in batch],
            "frame_indices": [item["frame_id"] for item in batch],
        }


class SimpleVideoCollectionLoader(VideoCollectionDataLoader):
    def __init__(
        self,
        videos: list[Path | str | Video],
        *,
        transform: Callable | None = None,
        buffer_size: int = 64,
        frame_id_regex: str | re.Pattern | None = r"frame\D*(\d+)(?!\d)",
        use_cached_video_metadata: bool = True,
        n_frame_counting_workers: int = -1,
        progress_bar: bool | None = None,
        min_frames_per_worker: int = 300,
        **kwargs,
    ):
        """Easier API for parallel video loading if you don't mind deviating from the
        standard "Dataset + DataLoader" pattern with torch.utils.data. Use this class
        like the DataLoader.

        The `videos` argument is a list of video specifications, each of which can be
        either an already-constructed Video object, or a path to video data. If it is a
        path, the Video object will be created automatically (specify arguments for the
        Video constructors here as keyword arguments).

        Other arguments are passed to VideoCollectionDataset and DataLoader (see their
        documentation).

        Once constructed, you can use an object of this class like a regular
        torch.utils.data.DataLoader, e.g. in a for loop over batches."""
        logger.info(
            "Checking requested videos and creating Video objects from paths if needed"
        )
        video_objects = self._resolve_videos(
            videos,
            buffer_size=buffer_size,
            frame_id_regex=frame_id_regex,
            use_cached_video_metadata=use_cached_video_metadata,
        )

        logger.info(f"Creating VideoCollectionDataset with {len(video_objects)} videos")
        dataset = VideoCollectionDataset(
            video_objects,
            transform=transform,
            use_cached_video_metadata=use_cached_video_metadata,
            n_frame_counting_workers=n_frame_counting_workers,
            progress_bar=progress_bar,
        )

        num_workers = kwargs.get("num_workers", 0)  # 0 is normal DataLoader default
        kwargs["num_workers"] = _resolve_n_workers_spec(num_workers)

        logger.info("Creating VideoCollectionDataLoader")
        super().__init__(dataset, min_frames_per_worker=min_frames_per_worker, **kwargs)

    @staticmethod
    def _resolve_videos(
        video_specs: list[Video | Path | str],
        buffer_size: int,
        frame_id_regex: str | re.Pattern | None,
        use_cached_video_metadata: bool,
    ) -> list[Video]:
        videos_resolved = []
        for video_spec in video_specs:
            if isinstance(video_spec, Video):
                video = video_spec
            elif isinstance(video_spec, (str, Path)):
                path = Path(video_spec).resolve()
                if path.is_dir():
                    logger.info(
                        f"Using ImageDirVideo backend for path {path}, which is a dir."
                    )
                    video = ImageDirVideo(path, frame_id_regex=frame_id_regex)
                elif path.is_file():
                    logger.info(
                        f"Using EncodedVideo backend for path {path}, which is a file."
                    )
                    video = EncodedVideo(
                        path,
                        buffer_size=buffer_size,
                        cache_metadata=True,  # will be cached anyway upon dataset init
                        use_cached_metadata=use_cached_video_metadata,
                    )
                else:
                    raise FileNotFoundError(f"Video path {path} does not exist.")
            else:
                raise TypeError(
                    f"Invalid video specification of type {type(video_spec)}. "
                    f"Expected Video, str, or Path."
                )
            videos_resolved.append(video)
        return videos_resolved


def _resolve_n_workers_spec(n_workers: int) -> int:
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
