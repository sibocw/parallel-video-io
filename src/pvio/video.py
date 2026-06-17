import torch
import logging
import re
import imageio.v2 as imageio
import numpy as np
from abc import abstractmethod, ABC
from time import time
from typing import Any, Callable
from torchcodec.decoders import VideoDecoder
from torch.utils.data import get_worker_info
from pathlib import Path
import os
import tempfile
import fcntl

from .io import get_video_metadata
from . import _accel


logger = logging.getLogger(__name__)


# A single per-user lock file used to serialize TorchCodec/FFmpeg decoder
# construction across all of this user's worker processes. See
# EncodedVideo._create_video_decoder for the rationale.
_DECODER_INIT_LOCK_PATH = (
    Path(tempfile.gettempdir()) / f"pvio_decoder_init.{os.getuid()}.lock"
)


class Video(ABC):
    def __init__(
        self,
        path: Path | str,
        frame_range: tuple[int, int] | None = None,
    ):
        r"""Base class for a video source (a file or a directory of images).

        Nothing is loaded at construction time. Call `.setup()` before reading frames.

        Args:
            path (Path | str): Path to the video source. Interpretation depends on
                the backend subclass.
            frame_range (tuple[int, int] | None): If specified, only frames in
                [start, end) are exposed. Defaults to the full video.
        """
        self.path = Path(path)
        self.frame_range_requested = frame_range

        self.__setup_done = False
        self.frame_range_effective: tuple[int, int] | None = None
        self.n_frames_in_source_video: int | None = None
        self.n_frames_in_range: int | None = None
        self.frame_size: tuple[int, int] | None = None
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
        """Validate `frame_range`; if None, return (0, n_frames_source_video)."""
        if requested_frame_range is not None:
            if (
                not isinstance(requested_frame_range, (tuple, list))
                or len(requested_frame_range) != 2
            ):
                raise ValueError(
                    f"frame_range must be a tuple of 2 integers, "
                    f"got {requested_frame_range!r}"
                )
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

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Validate arguments, load metadata, and prepare this video for reading.

        Must be called after ``__init__`` and before reading frames. Calling
        :meth:`read_frame` without a prior ``setup`` triggers an automatic
        setup with a warning.

        Args:
            *args: Forwarded to :meth:`_post_setup`.
            **kwargs: Forwarded to :meth:`_post_setup`.

        Raises:
            RuntimeError: If :meth:`_post_setup` reports failure.
        """
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

    def read_frame(self, index: int, transform: Callable | None = None) -> torch.Tensor:
        """Read a single frame as a CHW float32 tensor.

        Args:
            index: Virtual frame index; 0 is the start of the effective
                frame range.
            transform: Optional callable applied to the CHW float32 tensor
                before returning.

        Returns:
            Frame as a CHW float32 tensor with values in ``[0, 1]``.
        """
        if not self.__setup_done:
            logger.warning(
                f"Video at path {self.path} is not set up yet. Call `.setup()` first. "
                f"This might take some time, depending on the backend subclass."
            )
            self.setup()

        return self._read_frame(index, transform=transform)

    # THE FOLLOWING METHODS **MUST** BE IMPLEMENTED BY BACKEND SUBCLASSES
    @abstractmethod
    def _validate_init_params(self) -> None:
        """Validate the path for this backend. Raise on any error; return None."""
        pass

    @abstractmethod
    def _load_metadata(self) -> tuple[int, tuple[int, int], float]:
        """Return frame count, frame size, and FPS for the video source.

        Returns:
            A 3-tuple ``(n_frames_total, (height, width), fps)``.
            *n_frames_total* is the full frame count of the source, unclipped
            by *frame_range*. *fps* may be ``None`` if not applicable (e.g.
            for image directories).
        """
        pass

    @abstractmethod
    def _read_frame(
        self, index: int, transform: Callable | None = None
    ) -> torch.Tensor:
        """Return the frame at virtual index `index` as a CHW float tensor in [0, 1].
        Apply `transform` after loading if provided.

        `index` 0 corresponds to the start of the effective frame range."""
        pass

    # THE FOLLOWING METHODS CAN BE **OPTIONALLY** IMPLEMENTED BY BACKEND SUBCLASSES
    def _post_setup(self, *args: Any, **kwargs: Any) -> bool:
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
        device: str | None = None,
    ):
        """Video backend for "real" video files (e.g., mp4, mkv, etc.) using TorchCodec.

        Args:
            path (Path | str): Path to the video file.
            frame_range (tuple[int, int] | None): If specified, only frames in this
                range [start, end) are considered part of the video.
            buffer_size (int): Frames to decode in one batch. Larger values reduce
                decoding overhead at the cost of memory. Optimal size depends on
                keyframe intervals and available RAM; 64 (default) works well in most
                cases.
            cache_metadata (bool): Whether to cache video metadata to disk for faster
                subsequent metadata reads.
            use_cached_metadata (bool): Whether to use cached metadata when available.
                Set to False to force re-load metadata.
            device (str | None): Decode device for TorchCodec. ``None`` (default)
                auto-selects ``"cuda"`` when a CUDA GPU is available and ``"cpu"``
                otherwise; pass ``"cpu"`` or ``"cuda"`` to force a choice. Exact
                (frame-accurate) seeking is preserved on either device. GPU
                decoding returns frames already resident in GPU memory. Note that
                CUDA cannot be initialised inside forked DataLoader worker
                processes, so when this object is iterated under
                ``num_workers > 0`` the decoder automatically downgrades to CPU in
                the workers; use the single-process path (``num_workers=0``, as
                ``SimpleVideoCollectionLoader`` arranges automatically) to keep
                decoding on the GPU.
        """
        super().__init__(path, frame_range)
        self.buffer_size = buffer_size
        self.cache_metadata = cache_metadata
        self.use_cached_metadata = use_cached_metadata
        self.device = _accel.resolve_decode_device(device)

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
            self._decoder = self._create_video_decoder(self.path, self.device)

        vir_frame_id = index  # `index` is the virtual frame_id - make alias for clarity

        # If requested frame is already in buffer, apply transform and return
        if vir_frame_id in self._buffer:
            frame = self._buffer[vir_frame_id]
            if transform is not None:
                frame = transform(frame)
            return frame

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
        batch_frames = self._decode_buffer(phy_frame_ids_to_buffer)  # NCHW uint8
        batch_frames = batch_frames.float() / 255.0  # normalize to [0, 1]
        for i, _vfid in enumerate(vir_frame_ids_to_buffer):
            self._buffer[_vfid] = batch_frames[i, ...]  # store pre-transform

        frame = self._buffer[vir_frame_id]
        if transform is not None:
            frame = transform(frame)
        return frame

    def _decode_buffer(self, phy_frame_ids: np.ndarray) -> torch.Tensor:
        """Decode a batch of physical frame ids, with a GPU-OOM safety net.

        Decoding a buffer of large frames on the GPU can exhaust device memory
        (e.g. a 64-frame buffer of 4K frames is several GB). When that happens,
        permanently switch this video to CPU decoding and retry, rather than
        crashing — throughput drops for this video but results are still
        produced. Non-OOM errors propagate unchanged.
        """
        try:
            return self._decoder.get_frames_at(phy_frame_ids).data  # NCHW uint8
        except torch.cuda.OutOfMemoryError:
            if not str(self.device).startswith("cuda"):
                raise
            logger.warning(
                "GPU ran out of memory decoding %s (buffer_size=%d at this "
                "resolution is too large for device memory); falling back to CPU "
                "decoding for this video. Reduce buffer_size to keep GPU decoding.",
                self.path,
                self.buffer_size,
            )
            self._decoder = None
            torch.cuda.empty_cache()
            self.device = "cpu"
            self._decoder = self._create_video_decoder(self.path, "cpu")
            return self._decoder.get_frames_at(phy_frame_ids).data

    @staticmethod
    def _create_video_decoder(video_path: Path, device: str = "cpu") -> VideoDecoder:
        """Create a TorchCodec VideoDecoder while holding a shared init lock.

        Historically, constructing decoders concurrently across worker processes
        occasionally segfaulted (see PR #7). FFmpeg builds global state (e.g. codec
        lookup tables) on first use; per the libav-user thread below, that
        initialisation is thread-unsafe and must be serialized *within* a process —
        across separate processes the globals are not shared, so the original
        cross-process explanation does not actually hold. On modern FFmpeg (>=4.0;
        this project pins 6.x) codec registration is a no-op / thread-safe, and the
        crash is no longer reproducible: a stress test of ~2.5k maximally-concurrent
        constructions on the pinned stack produced zero crashes.

        The lock is kept only as cheap, defensive insurance for older FFmpeg builds.
        A single per-user lock file (`_DECODER_INIT_LOCK_PATH`) serializes *all*
        decoder construction across this user's worker processes — unlike the
        previous per-instance lock, which never serialized workers decoding
        different videos. It is held only during construction; frame reads run fully
        in parallel. See
        https://ffmpeg.org/pipermail/libav-user/2014-August/007298.html

        ``device`` selects CPU or CUDA (NVDEC) decoding; ``seek_mode="exact"`` keeps
        frame-accurate seeking on either. CUDA cannot be initialised inside a forked
        DataLoader worker, so a ``"cuda"`` request is downgraded to ``"cpu"`` when
        running in a worker subprocess. As a final safety net, a failed GPU decoder
        construction falls back to CPU rather than propagating the error."""
        if device.startswith("cuda") and get_worker_info() is not None:
            # Forked DataLoader workers cannot (re)initialise CUDA. Decode on CPU
            # here; the GPU path is intended for single-process iteration.
            logger.debug(
                "Running inside a DataLoader worker; decoding %s on CPU instead of %s.",
                video_path,
                device,
            )
            device = "cpu"

        with open(_DECODER_INIT_LOCK_PATH, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                decoder = VideoDecoder(
                    video_path.as_posix(),
                    seek_mode="exact",
                    dimension_order="NCHW",
                    device=device,
                )
            except Exception as e:
                if device == "cpu":
                    raise
                logger.warning(
                    "GPU decoder construction failed for %s on device %r (%s); "
                    "falling back to CPU decoding.",
                    video_path,
                    device,
                    e,
                )
                decoder = VideoDecoder(
                    video_path.as_posix(),
                    seek_mode="exact",
                    dimension_order="NCHW",
                    device="cpu",
                )
        return decoder

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
                range [start, end) are exposed. Note: start/end index the **sorted
                position** of the frames (0-based), not the physical frame id parsed
                from the filename. So with files whose parsed ids are 0, 5, 10, 15, 20,
                ``frame_range=(1, 3)`` exposes the 2nd and 3rd files (parsed ids 5 and
                10), matching how `EncodedVideo` treats frame_range as a positional
                slice. The parsed ids are used only for ordering and remain accessible
                via `frame_id_vir2phy` / `phy_frame_id_to_path`.
            frame_id_regex (str | re.Pattern | None): Regex used to parse the frame
                index from each filename (must capture exactly one group). The default
                handles names like "frame1.jpg", "frame002.tif", "frame_3.png",
                "frame-4.bmp", "frame_id=05.ext". If None, files are sorted
                alphabetically and indices are assigned 0, 1, 2, …
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

    def _post_setup(self, *args: Any, **kwargs: Any) -> bool:
        """Build the virtual/physical frame-id ↔ path mappings.
        Called after frame_range_effective is resolved so the mapping can be
        restricted to the requested range.

        frame_range is applied to the **sorted position** of each file (the
        enumeration index below), not to the parsed physical frame id. Both the
        regex and no-regex branches use this positional convention, so a frame_range
        always selects a contiguous slice of the ordered sequence. See the class
        docstring for the rationale."""
        all_paths = [path for path in self.path.iterdir() if path.is_file()]
        start, end = self.frame_range_effective

        if self.frame_id_regex is None:
            all_paths.sort(key=lambda f: f.name)
            for i, img_path in enumerate(all_paths):
                self.phy_frame_id_to_path[i] = img_path
                if start <= i < end:
                    vir_frame_id = i - start
                    self.vir_frame_id_to_path[vir_frame_id] = img_path
                    self.frame_id_vir2phy[vir_frame_id] = i
                    self.frame_id_phy2vir[i] = vir_frame_id
        else:
            phy_frame_id_and_path = [
                (self._parse_frame_id_from_filename(p.name, self.frame_id_regex), p)
                for p in all_paths
            ]
            phy_frame_id_and_path.sort(key=lambda x: x[0])
            for sorted_idx, (phy_frame_id, img_path) in enumerate(
                phy_frame_id_and_path
            ):
                self.phy_frame_id_to_path[phy_frame_id] = img_path
                if start <= sorted_idx < end:
                    vir_frame_id = sorted_idx - start
                    self.vir_frame_id_to_path[vir_frame_id] = img_path
                    self.frame_id_vir2phy[vir_frame_id] = phy_frame_id
                    self.frame_id_phy2vir[phy_frame_id] = vir_frame_id

        return True  # mark success

    def _load_metadata(self) -> tuple[int, tuple[int, int], float]:
        all_files = [f for f in self.path.iterdir() if f.is_file()]
        n_frames_total = len(all_files)  # n total images in dir (not just in range)
        if n_frames_total == 0:
            raise ValueError(f"No image files found in directory {self.path}.")

        # Note: frame_size is sampled from an arbitrary file because iterdir() returns
        # files in OS-defined order. For well-formed datasets all frames share the same
        # size, so this is fine in practice; if sizes differ the reported frame_size may
        # not reflect the majority or the first frame by sort order.
        sample_frame = imageio.imread(all_files[0])
        frame_size = sample_frame.shape[:2]

        fps = None  # not applicable for image directories

        return n_frames_total, frame_size, fps

    def _read_frame(self, index: int, transform: Callable | None) -> torch.Tensor:
        vir_frame_id = index  # `index` is the virtual frame_id - make alias for clarity
        if vir_frame_id not in self.vir_frame_id_to_path:
            raise IndexError(f"Frame index {index} out of bounds")
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
