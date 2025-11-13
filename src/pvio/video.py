import torch
import logging
import re
import imageio.v2 as imageio
import numpy as np
from abc import abstractmethod, ABC
from time import time
from typing import Callable
from torchcodec.decoders import VideoDecoder
from pathlib import Path

from .io import get_video_metadata


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

        # Run-time check of whether arguments to `__init__` are valid
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
        """Read a single frame at the specified index. A transform can be supplied to
        modify the frame after reading.

        Note: `index` here is the **virtual** frame index, i.e. the index is 0 at the
        start of the effective frame range.

        This is a wrapper around the actual `._read_frame()` method implemented by the
        backend subclass."""
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

        IMPORTANT: `index` here is the **virtual** frame index, i.e. the index is 0 at
        the start of the effective frame range.

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

        vir_frame_id = index  # `index` is the virtual frame_id - make alias for clarity

        # If requested frame is already in buffer, return it directly
        if vir_frame_id in self._buffer:
            return self._buffer[vir_frame_id]

        # Buffer has expired - expunge & refill
        # (loading many frames at once reduces decoding overhead)
        self._buffer.clear()
        buffer_vir_frame_id_limit = min(
            vir_frame_id + self.buffer_size, self.n_frames_in_range
        )
        vir_frame_ids_to_buffer = np.arange(vir_frame_id, buffer_vir_frame_id_limit)
        phy_frame_ids_to_buffer = (
            vir_frame_ids_to_buffer + self.frame_range_effective[0]
        )
        batch_frames = self._decoder.get_frames_at(phy_frame_ids_to_buffer).data  # NCHW
        batch_frames = batch_frames.float() / 255.0  # normalize to [0, 1]
        for i, _vfid in enumerate(vir_frame_ids_to_buffer):
            frame = batch_frames[i, ...]
            if transform is not None:
                frame = transform(frame)
            self._buffer[_vfid] = frame

        return self._buffer[vir_frame_id]

    def close(self):
        # VideoDecoder from torchcodec doesn't have a "close" method - just let Python
        # garbage collection handle it
        self._decoder = None


class ImageDirVideo(Video):
    def __init__(
        self,
        path: Path | str,
        frame_range: tuple[int, int] | None = None,
        frame_id_regex: str | re.Pattern | None = r"frame\D*(\d+)(?!\d)",
    ):
        r"""Video backend for directories containing frames as individual images.

        Args:
            path (Path | str): Path to the directory containing frames.
            frame_range (tuple[int, int] | None): If specified, only frames in this
                range [start, end) are considered part of the video.
            frame_id_regex (str | re.Pattern | None): For each frame file, the frame
                index is parsed from the filename using this regular expression. The
                default r"frame\D*(\d+)(?!\d)" is a pretty powerful one: it can broadly
                handle filenames like "frame1.jpg", "frame002.tif", "frame_3.png",
                "frame-4.bmp", "frame_id=05.custom.suffix". Use an online tool like
                https://regex101.com/ to test your regex patterns. If None, filenames
                are sorted alphabetically and frame indices are assigned from 0.
        """
        super().__init__(path, frame_range)
        if frame_id_regex is not None and isinstance(frame_id_regex, str):
            frame_id_regex = re.compile(frame_id_regex)
        self.frame_id_regex: re.Pattern | None = frame_id_regex

        # The following mappings are to be populated in `._post_setup()`
        self.phy_frame_id_to_path: dict[int, Path] = {}
        self.vir_frame_id_to_path: dict[int, Path] = {}
        self.frame_id_vir2phy: dict[int, int] = {}
        self.frame_id_phy2vir: dict[int, int] = {}

    @staticmethod
    def _parse_frame_id_from_filename(
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
        # Index frame paths by frame_ids
        all_paths = [path for path in self.path.iterdir() if path.is_file()]
        if self.frame_id_regex is None:
            all_paths.sort(key=lambda f: f.name)
            for i, img_path in enumerate(all_paths):
                self.phy_frame_id_to_path[i] = img_path
                self.vir_frame_id_to_path[i] = img_path
                self.frame_id_vir2phy[i] = i
                self.frame_id_phy2vir[i] = i
        else:
            phy_frame_id_and_path = [
                (self._parse_frame_id_from_filename(p.name, self.frame_id_regex), p)
                for p in all_paths
            ]
            phy_frame_id_and_path.sort(key=lambda x: x[0])
            for vir_frame_id, (phy_frame_id, img_path) in enumerate(
                phy_frame_id_and_path
            ):
                self.phy_frame_id_to_path[phy_frame_id] = img_path
                self.vir_frame_id_to_path[vir_frame_id] = img_path
                self.frame_id_vir2phy[vir_frame_id] = phy_frame_id
                self.frame_id_phy2vir[phy_frame_id] = vir_frame_id

        return True  # mark success

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
        vir_frame_id = index  # `index` is the virtual frame_id - make alias for clarity
        assert (
            vir_frame_id in self.vir_frame_id_to_path
        ), f"Frame index {index} not found"
        frame_path = self.vir_frame_id_to_path[vir_frame_id]
        frame = imageio.imread(frame_path)
        frame = torch.from_numpy(frame)
        if frame.ndim == 2:
            frame = frame.unsqueeze(-1)  # grayscale image, add channel dim
        frame = frame.permute(2, 0, 1)  # HWC to CHW
        frame = frame.float() / 255.0  # normalize to [0, 1]
        if transform is not None:
            frame = transform(frame)
        return frame
