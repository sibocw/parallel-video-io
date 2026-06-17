import numpy as np
import json
import logging
import os
import tempfile
import contextlib
import imageio.v2 as imageio
from pathlib import Path

from . import _accel


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _imageio_ffmpeg_exe(exe: str | None):
    """Temporarily point imageio-ffmpeg at *exe* via ``IMAGEIO_FFMPEG_EXE``.

    Used to route NVENC encodes through a system FFmpeg, since the binary
    bundled with imageio-ffmpeg is usually built without NVENC. A no-op when
    *exe* is None. Restores the previous environment on exit.
    """
    if exe is None:
        yield
        return
    sentinel = object()
    prev = os.environ.get("IMAGEIO_FFMPEG_EXE", sentinel)
    os.environ["IMAGEIO_FFMPEG_EXE"] = exe
    try:
        yield
    finally:
        if prev is sentinel:
            os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
        else:
            os.environ["IMAGEIO_FFMPEG_EXE"] = prev


def read_frames_from_video(
    video_path: Path | str, frame_indices: list[int] | None = None
) -> tuple[list[np.ndarray], float | None]:
    """Read specific frames from a video file.

    Args:
        video_path: Path to the video file.
        frame_indices: Frame indices to read. If ``None``, reads all frames.

    Returns:
        A 2-tuple ``(frames, fps)``. *frames* is a list of uint8 numpy arrays
        in ``(H, W, C)`` format. *fps* is the FPS reported by the container,
        or ``None`` if unavailable.
    """
    frames = []
    with imageio.get_reader(video_path) as reader:
        if frame_indices is None:
            frame_indices = list(range(reader.count_frames()))
        for frame_id in frame_indices:
            frames.append(reader.get_data(frame_id))
        fps = reader.get_meta_data().get("fps", None)
    return frames, fps


def write_frames_to_video(
    video_path: Path | str,
    frames: list[np.ndarray],
    fps: float,
    codec: str | None = None,
    ffmpeg_params: list[str] | None = None,
    log_interval: int | None = None,
) -> None:
    """Write a sequence of frames to a video file.

    By default the codec is chosen automatically: on a machine with a CUDA GPU
    and an NVENC-capable FFmpeg, frames are encoded with the GPU (H.264 NVENC at
    a visually-lossless setting, comparable to libx264 CRF 20); otherwise they
    fall back to libx264 on the CPU. The auto path also falls back to libx264 if
    an NVENC encode fails for any reason (e.g. frames below NVENC's minimum
    size), so output is always produced.

    Args:
        video_path: Path for the output video file.
        frames: Frames as uint8 numpy arrays in ``(H, W, C)`` format. All
            frames must share the same spatial dimensions.
        fps: Frames per second of the output video.
        codec: FFmpeg codec name. If ``None`` (default), the codec is selected
            automatically (NVENC on GPU, else libx264). Pass an explicit codec
            (e.g. ``"libx264"``, ``"h264_nvenc"``) to override auto-selection.
        ffmpeg_params: Raw FFmpeg parameter list. When ``None`` the parameters
            matching the selected codec are used (CRF 20 / slow / high profile
            for libx264; visually-lossless constant-QP for NVENC). Passing this
            explicitly while leaving *codec* as ``None`` keeps the libx264
            defaults (GPU auto-selection is disabled, since custom parameters
            are codec-specific). CRF 20 is more conservative than FFmpeg's
            default of 23, appropriate for scientific data where quality loss
            should be minimal; lower values produce higher quality at the cost
            of larger files.
        log_interval: If set, log progress every *log_interval* frames at
            ``INFO`` level.

    Raises:
        ValueError: If *frames* is empty or contains frames with mismatched
            dimensions.
    """
    # Check frame size consistency
    if len(frames) == 0:
        raise ValueError("No frames provided to write_frames_to_video")
    frame_size = frames[0].shape[:2]
    for frame in frames:
        if frame.shape[:2] != frame_size:
            raise ValueError(
                "All frames must have the same dimensions. The 0th frame has size "
                f"{frame_size}, but at least one frame has size {frame.shape[:2]}."
            )
    height, width = frame_size[0], frame_size[1]

    # Resolve the encode plan as a list of (codec, params, ffmpeg_exe) attempts.
    # The auto path tries NVENC first (when usable) and always keeps a libx264
    # fallback so output is produced even if the GPU encode fails.
    attempts: list[tuple[str, list[str], str | None]] = []
    if codec is not None:
        # Explicit codec: honour it verbatim, no auto-selection or fallback.
        params = ffmpeg_params if ffmpeg_params is not None else _accel.LIBX264_PARAMS
        attempts.append((codec, params, None))
    elif ffmpeg_params is not None:
        # Custom params but no codec: parameters are codec-specific, so stay on
        # the libx264 default rather than silently switching to NVENC.
        attempts.append((_accel.LIBX264_CODEC, ffmpeg_params, None))
    else:
        # Pure auto: NVENC when available, with a libx264 fallback.
        if _accel.can_use_nvenc(height, width):
            attempts.append(
                (_accel.NVENC_CODEC, _accel.NVENC_PARAMS, _accel.nvenc_ffmpeg_exe())
            )
        attempts.append((_accel.LIBX264_CODEC, _accel.LIBX264_PARAMS, None))

    last_error: Exception | None = None
    for attempt_idx, (attempt_codec, attempt_params, ffmpeg_exe) in enumerate(attempts):
        is_last = attempt_idx == len(attempts) - 1
        try:
            with _imageio_ffmpeg_exe(ffmpeg_exe):
                _encode_frames(
                    video_path, frames, fps, attempt_codec, attempt_params, log_interval
                )
            return
        except Exception as e:
            last_error = e
            if is_last:
                raise
            logger.warning(
                "Encoding with codec %r failed (%s); falling back to %r.",
                attempt_codec,
                e,
                attempts[attempt_idx + 1][0],
            )
    # Unreachable: the last attempt either returns or re-raises.
    if last_error is not None:  # pragma: no cover - defensive
        raise last_error


def _encode_frames(
    video_path: Path | str,
    frames: list[np.ndarray],
    fps: float,
    codec: str,
    ffmpeg_params: list[str],
    log_interval: int | None,
) -> None:
    """Encode *frames* to *video_path* with imageio's FFmpeg backend."""
    with imageio.get_writer(
        str(video_path),
        "ffmpeg",
        fps=fps,
        codec=codec,
        quality=None,  # Use CRF/QP (in ffmpeg_params) instead of quality
        ffmpeg_params=ffmpeg_params,
    ) as video_writer:
        for i, frame in enumerate(frames):
            video_writer.append_data(frame)

            if log_interval is not None and (i + 1) % log_interval == 0:
                logger.info(f"Written frame {i + 1}/{len(frames)}")


def check_num_frames(video_path: Path | str) -> int:
    """Return the number of frames in a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Total frame count.

    Raises:
        RuntimeError: If the file cannot be opened.
    """
    try:
        with imageio.get_reader(video_path) as reader:
            num_frames = reader.count_frames()
    except Exception as e:
        raise RuntimeError(f"Failed to open video file: {video_path}") from e
    return num_frames


def get_video_metadata(
    video_path: Path | str,
    cache_metadata: bool = True,
    use_cached_metadata: bool = True,
    metadata_suffix: str = ".metadata.json",
) -> dict[str, int | tuple[int, int] | float | None]:
    """Return frame count, frame size, and FPS for a video file.

    Results are cached to a sidecar JSON file alongside the video to avoid
    re-reading on subsequent calls.

    Args:
        video_path: Path to the video file.
        cache_metadata: Write metadata to a cache file after reading.
        use_cached_metadata: Return cached metadata if the sidecar file
            exists. Set to ``False`` to force a fresh read.
        metadata_suffix: Suffix appended to the video filename to form the
            cache path. Default: ``".metadata.json"``.

    Returns:
        Dictionary with keys ``"n_frames"`` (int total frame count),
        ``"frame_size"`` (tuple ``(height, width)``), and ``"fps"``
        (float or ``None`` if unavailable).
    """
    video_path = Path(video_path)
    cache_path = video_path.parent / (video_path.name + metadata_suffix)
    metadata = {}
    if use_cached_metadata and cache_path.is_file():
        try:
            with open(cache_path, "r") as f:
                metadata = json.load(f)
            n_frames = metadata["n_frames"]
            frame_size = tuple(metadata["frame_size"])
            fps = metadata["fps"]
        except Exception as e:
            logger.critical(f"Corrupted metadata cache file {cache_path}: {e}")
            raise
    else:
        n_frames = check_num_frames(video_path)
        sample_frames, fps = read_frames_from_video(video_path, frame_indices=[0])
        frame_size = sample_frames[0].shape[:2]

        if cache_metadata:
            metadata = {
                "n_frames": n_frames,
                "frame_size": list(frame_size),
                "fps": fps,
            }
            with tempfile.NamedTemporaryFile(
                mode="w", dir=cache_path.parent, suffix=".tmp", delete=False
            ) as tmp_f:
                tmp_path = tmp_f.name
                json.dump(metadata, tmp_f, indent=2)
            os.replace(tmp_path, cache_path)

    return {"n_frames": n_frames, "frame_size": frame_size, "fps": fps}
