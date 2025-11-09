import torch
import logging
import re
import imageio.v2 as imageio
import numpy as np
from abc import abstractmethod, ABC
from sys import stderr
from multiprocessing import cpu_count
from time import time
from typing import Callable
from torchcodec.decoders import VideoDecoder
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from .video_io import get_video_metadata


logger = logging.getLogger(__name__)


class Video(ABC):
    def __init__(
        self,
        path: Path | str,
        frame_range: tuple[int, int] | None = None,
    ):
        r"""Encapsulate a "video," which can be either a video file or a directory
        containing individual image files. You can also specify a range of frames, which
        allows you to treat a subset of the video as a separate video.

        NOTE: Upon `__init__`, nothing is actually loaded yet. The object only contains
        a specification of where the video is and what to read from it. It is required to
        call `.setup()` after init to load metadata and prepare for reading.

        Args:
            path (Path | str): Video path, exactly what it is depends on the backend.
            frame_range (tuple[int, int] | None): If specified, only frames in this
                range [start, end) are considered part of the video.
        """
        self.path = Path(path)
        self.frame_range_requested = frame_range

        self.__setup_done = False
        self.frame_range_effective: tuple[int, int] = None
        self.n_frames_in_source_video: int = None
        self.n_frames_in_range: int = None
        self.frame_size: tuple[int, int] = None
        self.fps: float | None = None

    def __len__(self):
        if not self.__setup_done:
            raise RuntimeError(
                f"Video at path {self.path} is not set up yet. Call `.setup()` first."
            )
        return self.n_frames_in_range

    @staticmethod
    def _resolve_effective_frame_range(
        requested_frame_range: tuple[int, int] | None, n_frames_source_video: int
    ) -> tuple[int, int]:
        """Check if user-specified `frame_range` is valid. If it's set to None for
        automatic determination, do so using `n_frames_source_video`."""
        if requested_frame_range is not None:
            start, end = requested_frame_range
            if start < 0 or end > n_frames_source_video or start >= end:
                raise ValueError(
                    f"Invalid frame_range {requested_frame_range} for video with "
                    f"{n_frames_source_video} frames."
                )
            return (start, end)
        else:
            logger.debug(
                "No frame_range specified. Using the full range of the source video."
            )
            return (0, n_frames_source_video)

    def setup(self, *args, **kwargs) -> None:
        """This method is to be called after `__init__`. It first validates arguments
        given to `__init__`, loads metadata, and then calls the backend subclass's
        `._post_setup()` method with arguments passed to this method."""
        if self.__setup_done:
            logger.warning(
                f"Video at path {self.path} is already set up. Skipping redundant call."
            )
            return

        # Run-time check of whether arguments to `__init__`
        self._validate_init_params()

        # Load video metadata - this might take a non-negligible amount time depending
        # on the backend subclass
        n_frames_total, frame_size, fps = self._load_metadata()
        self.n_frames_in_source_video = n_frames_total
        self.frame_size = frame_size
        self.fps = fps

        # Resolve effective frame range
        resolved_range = self._resolve_effective_frame_range(
            self.frame_range_requested, n_frames_total
        )
        self.frame_range_effective = resolved_range
        self.n_frames_in_range = resolved_range[1] - resolved_range[0]

        # Post-setup hook for backend subclasses
        post_setup_success = self._post_setup(*args, **kwargs)
        if not post_setup_success:
            raise RuntimeError(
                f"Backend subclass {self.__class__.__name__} failed to complete "
                f"post-setup operations (`._post_setup()`)."
            )

        self.__setup_done = True

    def read_frame(
        self, index: int, transform: Callable | None = None, *args, **kwargs
    ) -> torch.Tensor:
        """Read a single frame at the specified index. This is a wrapper around the
        actual `._read_frame()` method implemented by the backend subclass. A transform
        can be supplied to modify the frame after reading."""
        if not self.__setup_done:
            logger.warning(
                f"Video at path {self.path} is not set up yet. Call `.setup()` first. "
                f"This might take some time, depending on the backend subclass."
            )
            self.setup()

        return self._read_frame(index, transform=transform, *args, **kwargs)

    # THE FOLLOWING METHODS **MUST** BE IMPLEMENTED BY BACKEND SUBCLASSES
    @abstractmethod
    def _validate_init_params(self) -> None:
        """Check if the provided path is valid for this type of video. Nothing is
        returned; raise exceptions instead if there's any issue."""
        pass

    @abstractmethod
    def _load_metadata(self) -> tuple[int, tuple[int, int], float]:
        """Load and return the following: (1) total number of frames in the source video
        (in the entire video, not just in the specified frame_range), (2) frame size as
        a tuple (height, width), and (3) fps as a float (or None if not applicable)."""
        pass

    @abstractmethod
    def _read_frame(
        self, index: int, transform: Callable | None = None, *args, **kwargs
    ) -> torch.Tensor:
        """Read a single frame at the specified index. An optional `transform` argument
        must be supported. Returns the frame as a torch Tensor.

        IMPORTANT: `index` here is the **virtual** frame index, i.e. it is counted from
        0 and relative to the start of the effective frame range. For example, if the
        actual video has 100 frames and frame_range=(10, 20), then index 0 corresponds
        to frame 10 of the actual video.

        This method will be wrapped by the public `.read_frame()` method.
        """
        pass

    # THE FOLLOWING METHODS CAN BE **OPTIONALLY** IMPLEMENTED BY BACKEND SUBCLASSES
    def _post_setup(self, *args, **kwargs) -> bool:
        """Optional backend-specific logic to be called at the end of `.setup()`.
        Should return True if setup is successful, False otherwise."""
        return True

    def close(self) -> None:
        """Release any resources held by this Video object."""
        pass


class EncodedVideo(Video):
    def __init__(
        self,
        path: Path | str,
        frame_range: tuple[int, int] | None = None,
        buffer_size: int = 64,
        cache_metadata: bool = True,
        use_cached_metadata: bool = True,
    ):
        """Video backend for "real" video files (e.g., mp4, mkv, etc.) using TorchCodec.

        Args:
            path (Path | str): Path to the video file.
            frame_range (tuple[int, int] | None): If specified, only frames in this
                range [start, end) are considered part of the video.
            buffer_size (int): Number of frames to buffer when reading from video files.
                Decoding frames one by one is very inefficient. Since we know we're
                likely to load many frames in sequence, we can decode a batch of frames
                at once and buffer them. The optimal buffer size depends on keypoint
                intervals in the video, RAM availability, etc., but 64 (default) is a
                sweet spot in most cases.
            cache_metadata (bool): Whether to cache video metadata to disk for faster
                subsequent metadata reads.
            use_cached_metadata (bool): Whether to use cached metadata when available.
                Set to False to force re-load metadata.
        """
        super().__init__(path, frame_range)
        self.buffer_size = buffer_size
        self.cache_metadata = cache_metadata
        self.use_cached_metadata = use_cached_metadata

        # The following are to be managed by `.read_frame()`
        self._decoder: VideoDecoder | None = None
        self._buffer: dict[int, torch.Tensor] = {}

    def _validate_init_params(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Video path {self.path} doesn't exist or is not a file."
            )

    def _load_metadata(self) -> tuple[int, tuple[int, int], float]:
        start_time = time()

        metadata = get_video_metadata(
            self.path,
            cache_metadata=self.cache_metadata,
            use_cached_metadata=self.use_cached_metadata,
        )
        n_frames_total = metadata["n_frames"]
        frame_size = metadata["frame_size"]
        fps = metadata["fps"]

        walltime = time() - start_time
        logger.debug(f"Loaded metadata for video {self.path} in {walltime:.2f}s.")

        return n_frames_total, frame_size, fps

    def _read_frame(self, index: int, transform: Callable | None) -> torch.Tensor:
        # If this is the first time reading from this video, initialize the decoder
        if self._decoder is None:
            self._decoder = VideoDecoder(
                self.path.as_posix(), seek_mode="exact", dimension_order="NCHW"
            )

        # If requested frame is already in buffer, return it directly
        # NOTE: the buffer always uses **virtual** frame IDs!
        if index in self._buffer:
            return self._buffer[index]

        # Otherwise, expunge & refill buffer
        # (load many frames at once to reduce decoding overhead)
        self._buffer.clear()
        idx_to_buffer_until = min(index + self.buffer_size, self.n_frames_in_range)
        buffer_virtual_frameids = np.arange(index, idx_to_buffer_until)
        buffer_real_frameids = buffer_virtual_frameids + self.frame_range_effective[0]
        batch_frames = self._decoder.get_frames_at(buffer_real_frameids).data  # NCHW
        batch_frames = batch_frames.float() / 255.0  # normalize to [0, 1]
        for i, virtual_idx in enumerate(buffer_virtual_frameids):
            frame = batch_frames[i, ...]
            if transform is not None:
                frame = transform(frame)
            self._buffer[virtual_idx] = frame

        return self._buffer[index]

    def close(self):
        # VideoDecoder from torchcodec doesn't have a "close" method - just let Python
        # garbage collection handle it
        self._decoder = None


class ImageDirVideo(Video):
    def __init__(
        self,
        path: Path | str,
        frame_range: tuple[int, int] | None = None,
        frameid_regex: str | re.Pattern | None = r"frame\D*(\d+)(?!\d)",
    ):
        r"""Video backend for directories containing frames as individual images.

        Args:
            path (Path | str): Path to the directory containing frames.
            frame_range (tuple[int, int] | None): If specified, only frames in this
                range [start, end) are considered part of the video.
            frameid_regex (str | re.Pattern | None): For each frame file, the frame
                index is parsed from the filename using this regular expression. The
                default r"frame\D*(\d+)(?!\d)" is a pretty powerful one: it can broadly
                handle filenames like "frame1.jpg", "frame002.tif", "frame_3.png",
                "frame-4.bmp", "frameid=05.custom.suffix". If None, filenames are sorted
                alphabetically and frame indices are assigned from 0.
        """
        super().__init__(path, frame_range)

        # Index frame paths by frameids
        self.frameid_to_path: dict[int, Path] = {}
        if frameid_regex is None:
            all_paths = [f for f in self.path.iterdir() if f.is_file()]
            all_paths.sort(key=lambda f: f.name)
            for i, path in enumerate(all_paths):
                self.frameid_to_path[i] = path
        else:
            if isinstance(frameid_regex, str):
                frameid_regex = re.compile(frameid_regex)
            _frameid_to_path = {
                self._parse_frameid_from_filename(f.name, frameid_regex): f
                for f in self.path.iterdir()
                if f.is_file()
            }
            self.frameid_to_path = {
                frameid: _frameid_to_path[frameid]
                for frameid in sorted(_frameid_to_path.keys())
            }

    @staticmethod
    def _parse_frameid_from_filename(
        filename: str, regex_pattern: re.Pattern | str
    ) -> int:
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

    def _validate_init_params(self) -> None:
        if not self.path.is_dir():
            raise FileNotFoundError(
                f"Video path {self.path} is not a valid directory. "
                f"A directory containing individual frame images is expected."
            )

    def _post_setup(self):
        """Verify that all frame indices in the effective frame range are present. This
        has to be implemented in _post_setup because frame range is not resolved until
        this point."""
        for index in range(*self.frame_range_effective):
            if index not in self.frameid_to_path:
                raise FileNotFoundError(
                    f"Frame range {self.frame_range_effective} includes index {index}, "
                    f"but frame index {index} is not found in image directory "
                    f"{self.path} (which contains {len(self.frameid_to_path)} frames)."
                )
        return True

    def _load_metadata(self) -> tuple[int, tuple[int, int], float]:
        all_files = [f for f in self.path.iterdir() if f.is_file()]
        n_frames_total = len(all_files)  # n total images in dir (not just in range)
        if n_frames_total == 0:
            raise ValueError(f"No image files found in directory {self.path}.")

        sample_frame = imageio.imread(all_files[0])
        frame_size = sample_frame.shape[:2]

        fps = None  # not applicable for image directories

        return n_frames_total, frame_size, fps

    def _read_frame(self, index: int, transform: Callable | None) -> torch.Tensor:
        physical_index = index + self.frame_range_effective[0]
        assert physical_index in self.frameid_to_path, f"Frame index {index} not found"
        frame_path = self.frameid_to_path[physical_index]
        frame = imageio.imread(frame_path)
        frame = torch.from_numpy(frame)
        if frame.ndim == 2:
            frame = frame.unsqueeze(-1)  # grayscale image, add channel dim
        frame = frame.permute(2, 0, 1)  # HWC to CHW
        frame = frame.float() / 255.0  # normalize to [0, 1]
        if transform is not None:
            frame = transform(frame)
        return frame


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
        # to the same actual video)
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

        # Initialize attributes for worker assignment and buffering during loading
        # worker assignment format: (video_idx, start_frameid, end_frameid). Start and
        # end frameids are counted from 0 - if the video object has a frame_range, this
        # frameid would be relative to the start of that range.
        # This will be properly initialized in `.assign_workers`
        self.worker_assignments: list[list[tuple[int, int, int]]] = []
        self._frames_buffer: dict[tuple[int, int], torch.Tensor] = {}

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
        # Make a giant array of (video_idx, frameid) pairs. frameid is counted from 0
        # and relative to video.frame_range_effective[0]
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
            start_idx = worker_id * n_frames_per_worker
            end_idx = min(start_idx + n_frames_per_worker, self.n_frames_total)
            frame_specs = frame_specs_all[start_idx:end_idx, :]
            # Convert to list of (video_idx, start_frameid, end_frameid)
            # start/end frame IDs are **virtual** indices relative to frame_range[0]
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
            video = self.videos[video_idx]
            for frame_idx in range(start_frame_idx, end_frame_idx):
                frame = video.read_frame(frame_idx, transform=self.transform)
                yield {"frame": frame, "video_idx": video_idx, "frame_idx": frame_idx}

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
            "video_indices": [item["video_idx"] for item in batch],
            "frame_indices": [item["frame_idx"] for item in batch],
        }


class SimpleVideoCollectionLoader(VideoCollectionDataLoader):
    def __init__(
        self,
        videos: list[Path | str | Video],
        *,
        transform: Callable | None = None,
        buffer_size: int = 64,
        frameid_regex: str | re.Pattern | None = r"frame\D*(\d+)(?!\d)",
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
            frameid_regex=frameid_regex,
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
        frameid_regex: str | re.Pattern | None,
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
                    video = ImageDirVideo(path, frameid_regex=frameid_regex)
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
