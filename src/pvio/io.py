import numpy as np
import json
import logging
import os
import tempfile
import imageio.v2 as imageio
from pathlib import Path


logger = logging.getLogger(__name__)


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
    codec: str = "libx264",
    ffmpeg_params: list[str] | None = None,
    log_interval: int | None = None,
) -> None:
    """Write a sequence of frames to a video file.

    Args:
        video_path: Path for the output video file.
        frames: Frames as uint8 numpy arrays in ``(H, W, C)`` format. All
            frames must share the same spatial dimensions.
        fps: Frames per second of the output video.
        codec: FFmpeg codec name. Default: ``"libx264"``.
        ffmpeg_params: Raw FFmpeg parameter list. If ``None``, uses
            high-quality H.264 defaults (CRF 20, slow preset, high profile).
            CRF 20 is more conservative than FFmpeg's default of 23, which is
            appropriate for scientific data where quality loss should be
            minimal. Lower values (e.g. 18) produce higher quality at the
            cost of larger file sizes.
        log_interval: If set, log progress every *log_interval* frames at
            ``INFO`` level.

    Raises:
        ValueError: If *frames* is empty or contains frames with mismatched
            dimensions.
    """
    if ffmpeg_params is None:
        ffmpeg_params = [
            "-crf",
            "20",  # Lower = higher quality; 20 is conservative vs FFmpeg's default 23
            "-preset",
            "slow",  # Slower preset = better compression efficiency
            "-profile:v",
            "high",  # Use high profile for better compression
            "-level",
            "4.0",  # H.264 level
        ]

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

    # Use imageio to write video with ffmpeg backend
    with imageio.get_writer(
        str(video_path),
        "ffmpeg",
        fps=fps,
        codec=codec,
        quality=None,  # Use CRF (in ffmpeg_params) instead of quality
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
