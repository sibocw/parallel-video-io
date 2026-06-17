import torch
import logging
import re
import numpy as np
from sys import stderr
from multiprocessing import cpu_count
from typing import Any, Callable
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
    ) -> None:
        r"""Iterable dataset that yields frames from a list of Video objects.

        Each frame is yielded as a CHW float32 tensor with values in [0, 1].
        Call `.assign_workers()` (done automatically by VideoCollectionDataLoader)
        before iterating.

        Args:
            videos (list[Video]): Video objects to iterate. Videos are set up here;
                do not call `.setup()` on them beforehand.
            transform (Callable | None): Applied to each tensor before yielding.
                The following are already applied before the transform: (i)
                numpy array → torch tensor, (ii) HWC → CHW layout, and
                (iii) uint8 ``[0, 255]`` → float32 ``[0, 1]``.
            use_cached_video_metadata (bool): Use cached metadata for EncodedVideo
                objects if available. Set to False to force re-reading.
            n_frame_counting_workers (int): Parallel workers for pre-loading
                EncodedVideo metadata. -1 uses all available cores.
            progress_bar (bool | None): Show a progress bar during metadata loading.
                If None, shows one only when stderr is a TTY.
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
        """Distribute frames across workers for balanced parallel loading.

        Frames are assigned in contiguous ranges within each video to minimise
        seeking overhead. If the per-worker frame count would fall below
        `min_frames_per_worker`, the effective worker count is reduced accordingly.

        Args:
            n_loading_workers (int): Number of workers to distribute frames across.
            min_frames_per_worker (int): Lower bound on frames per worker. Workers
                below this threshold are merged to avoid excessive overhead.
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
        if start_global_frame_id != self.n_frames_total:
            raise RuntimeError(
                f"Frame index mismatch after building frame_specs: "
                f"got {start_global_frame_id}, expected {self.n_frames_total}"
            )
        if np.any(frame_specs_all == -1):
            raise RuntimeError("frame_specs_all contains uninitialized entries")

        # Dynamically balance load among workers
        n_frames_per_worker = int(np.ceil(self.n_frames_total / n_loading_workers))
        logger.info(
            f"Assigning {self.n_frames_total} total frames from "
            f"{len(self.videos)} videos to {n_loading_workers} loading workers."
        )
        n_workers_effective = n_loading_workers
        if n_frames_per_worker < min_frames_per_worker:
            n_frames_per_worker = min_frames_per_worker
            n_workers_effective = int(
                np.ceil(self.n_frames_total / n_frames_per_worker)
            )
            logger.info(
                f"`n_frames_per_worker` is less than `min_frames_per_worker` "
                f"({min_frames_per_worker}). This will result in many workers working "
                f"on not so much data, leading to high overhead. "
                f"Increasing `n_frames_per_worker` to {n_frames_per_worker} and "
                f"reducing `n_loading_workers` to {n_workers_effective}."
            )

        # Initialize worker assignments (original number of workers! Unused workers get
        # empty assignments)
        self.worker_assignments = [[] for _ in range(n_loading_workers)]

        for worker_id in range(n_workers_effective):
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
                # Cast to Python int: these flow into the yielded batch dicts, and the
                # public API documents video_indices/frame_indices as lists of int (not
                # numpy.int32, which would surprise strict isinstance checks / JSON).
                start_vir_frame_id = int(vir_frame_ids_local[0])
                end_vir_frame_id = int(vir_frame_ids_local[-1]) + 1  # exclusive
                self.worker_assignments[worker_id].append(
                    (int(video_id), start_vir_frame_id, end_vir_frame_id)
                )
        _nframes_total_check = 0
        for worker_chunks in self.worker_assignments:
            for _, start_vir_frame_id, end_vir_frame_id in worker_chunks:
                _nframes_total_check += end_vir_frame_id - start_vir_frame_id
        if _nframes_total_check != self.n_frames_total:
            raise RuntimeError(
                f"Frame count mismatch: assigned {_nframes_total_check} frames "
                f"but expected {self.n_frames_total}"
            )

    def __iter__(self):
        if not self.worker_assignments:
            raise RuntimeError(
                "VideoCollectionDataset requires VideoCollectionDataLoader. "
                "Did you forget to wrap the dataset in VideoCollectionDataLoader, "
                "or call .assign_workers() before iterating?"
            )

        # Get worker info for distributed loading
        worker_info = get_worker_info()
        if worker_info is None:
            # Single process
            if len(self.worker_assignments) != 1:
                raise RuntimeError(
                    "Using a single worker but worker assignments indicate multiple workers."
                )
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

    Each worker is assigned a contiguous range of frames to minimise seeking
    overhead. Batch dicts contain:

    - ``frames``: Tensor of shape (batch_size, C, H, W)
    - ``video_indices``: list of int — index into the videos list
    - ``frame_indices``: list of int — virtual frame index within that video

    Custom ``batch_sampler`` and ``collate_fn`` are not supported.
    """

    dataset: VideoCollectionDataset

    def __init__(
        self,
        dataset: VideoCollectionDataset,
        min_frames_per_worker: int = 300,
        **kwargs: Any,
    ) -> None:
        """Wrap a VideoCollectionDataset in a DataLoader with automatic worker assignment.

        Args:
            dataset (VideoCollectionDataset): The dataset to load from.
            min_frames_per_worker (int): Minimum frames per worker; the effective
                worker count is reduced if this threshold would otherwise be breached.
            **kwargs: Forwarded to torch.utils.data.DataLoader.
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
        frames = [item["frame"] for item in batch]
        # A collection may mix GPU-decoded videos (EncodedVideo on CUDA) with
        # CPU-decoded ones (e.g. ImageDirVideo), producing frames on different
        # devices that cannot be stacked directly. Promote everything to a CUDA
        # device when any frame is already there, keeping the batch GPU-resident.
        if len({f.device for f in frames}) > 1:
            target = next(
                (f.device for f in frames if f.device.type == "cuda"),
                frames[0].device,
            )
            frames = [f.to(target) for f in frames]
        return {
            "frames": torch.stack(frames),
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
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a VideoCollectionDataset and VideoCollectionDataLoader in one call.

        Each entry in *videos* may be a pre-constructed :class:`Video` object
        or a path (str / Path). Paths pointing to files become
        :class:`EncodedVideo`; paths pointing to directories become
        :class:`ImageDirVideo`.

        The decode workflow is selected automatically: on a machine with a CUDA
        GPU, file-backed videos decode on the GPU (NVDEC) and iteration runs in
        the main process (``num_workers`` is forced to 0, since CUDA cannot be
        used in forked workers); on a CPU-only machine, decoding uses the
        requested number of CPU workers as before. Pass ``device="cpu"`` to opt
        out and keep multi-worker CPU decoding even when a GPU is present.

        Args:
            videos: Video sources.
            transform: Applied to each CHW float32 frame tensor before it is
                yielded.
            buffer_size: Decode-buffer size forwarded to :class:`EncodedVideo`
                for file-path entries.
            frame_id_regex: Regex forwarded to :class:`ImageDirVideo` for
                directory-path entries.
            use_cached_video_metadata: Use cached metadata when available. Set
                to ``False`` to force fresh reads.
            n_frame_counting_workers: Workers for parallel metadata loading.
                ``-1`` uses all available cores.
            progress_bar: Show a progress bar during metadata loading.
                Defaults to ``True`` when stderr is a TTY.
            min_frames_per_worker: Minimum frames per worker; see
                :meth:`VideoCollectionDataset.assign_workers`.
            device: Decode device for file-backed videos, forwarded to
                :class:`EncodedVideo`. ``None`` (default) auto-selects the GPU
                when available; ``"cpu"``/``"cuda"`` force a choice.
            **kwargs: Forwarded to :class:`~torch.utils.data.DataLoader`.
        """
        logger.info(
            "Checking requested videos and creating Video objects from paths if needed"
        )
        video_objects = self._resolve_videos(
            videos,
            buffer_size=buffer_size,
            frame_id_regex=frame_id_regex,
            use_cached_video_metadata=use_cached_video_metadata,
            device=device,
        )

        # On the GPU decode path, iteration must run in the main process: CUDA
        # cannot be initialised in forked DataLoader workers. If any file-backed
        # video will decode on CUDA, force num_workers=0 so frames are decoded on
        # the GPU rather than silently downgraded to CPU in worker subprocesses.
        gpu_decode = any(
            isinstance(v, EncodedVideo) and str(v.device).startswith("cuda")
            for v in video_objects
        )
        if gpu_decode:
            requested_workers = kwargs.get("num_workers", 0)
            if requested_workers not in (0, None):
                logger.info(
                    "GPU decoding selected; forcing num_workers=0 so iteration runs "
                    "in the main process (was %s). CUDA cannot be used in forked "
                    "DataLoader workers. Pass device='cpu' to keep CPU multi-worker "
                    "loading.",
                    requested_workers,
                )
            kwargs["num_workers"] = 0

        logger.info(f"Creating VideoCollectionDataset with {len(video_objects)} videos")
        dataset = VideoCollectionDataset(
            video_objects,
            transform=transform,
            use_cached_video_metadata=use_cached_video_metadata,
            n_frame_counting_workers=n_frame_counting_workers,
            progress_bar=progress_bar,
        )

        num_workers = kwargs.get("num_workers", 0)  # 0 is normal DataLoader default
        if num_workers != 0:
            kwargs["num_workers"] = _resolve_n_workers_spec(num_workers)
        # num_workers=0 is passed through as-is; VideoCollectionDataLoader handles it by
        # running in the main process while still calling assign_workers(1)

        logger.info("Creating VideoCollectionDataLoader")
        super().__init__(dataset, min_frames_per_worker=min_frames_per_worker, **kwargs)

    @staticmethod
    def _resolve_videos(
        video_specs: list[Video | Path | str],
        buffer_size: int,
        frame_id_regex: str | re.Pattern | None,
        use_cached_video_metadata: bool,
        device: str | None = None,
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
                        device=device,
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
    """Resolve a worker-count spec to a concrete positive integer.

    Negative values follow the joblib convention: ``-1`` means all cores,
    ``-2`` means all but one, and so on. ``0`` is mapped to ``1`` because
    main-process iteration is not implemented.

    Args:
        n_workers: Worker-count spec.

    Returns:
        Resolved worker count (always ≥ 1).

    Raises:
        ValueError: If *n_workers* is less than ``-n_cpu_cores``.
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
            "n_workers_spec=0 interpreted as 1 worker. Processing in main thread not "
            "implemented; will use a single worker process/thread instead."
        )
        return 1
    elif n_workers > 0:
        return n_workers
    else:
        pass  # cannot resolve (n_workers < -n_cpu_cores)

    if n_workers_resolved is None:
        raise ValueError(
            f"Invalid n_workers_spec={n_workers}. Must be >= {-n_cpu_cores} "
            f"(n_cpu_cores={n_cpu_cores})."
        )
    return n_workers_resolved
