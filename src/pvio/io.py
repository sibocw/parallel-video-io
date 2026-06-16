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
        video_path (Path | str): Path to the video file.
        frame_indices (list[int] | None): Indices to read. If None, reads all frames.

    Returns:
        frames (list[np.ndarray]): Frames as uint8 numpy arrays in [height, width,
            channels] format.
        fps (float | None): FPS reported by the container, or None if unavailable.
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
        video_path (Path | str): Path to save the video file.
        frames (list[np.ndarray]): List of frames as numpy arrays (in
            [height, width, channels] format).
        fps (float): Frames per second for the output video.
        codec (str): Codec to use. Default: 'libx264'.
        ffmpeg_params (list[str] | None): Raw ffmpeg parameter list. If None (default),
            uses high-quality H.264 defaults (CRF 15, slow preset, high profile).
        log_interval (int | None): If set, log progress every `log_interval` frames at
            INFO level.
    """
    if ffmpeg_params is None:
        ffmpeg_params = [
            "-crf",
            "15",  # Lower CRF = higher quality (15 is very high quality)
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

            if log_interval is not None and i % log_interval == 0:
                logger.info(f"Written frame {i + 1}/{len(frames)}")


def check_num_frames(video_path: Path | str) -> int:
    """Check number of frames in a video file."""
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

    Results are cached to a sidecar file (video stem + metadata_suffix) to avoid
    re-reading on subsequent calls.

    Args:
        video_path (Path | str): Path to the video file.
        cache_metadata (bool): Write metadata to a cache file after reading.
        use_cached_metadata (bool): Return cached metadata if the sidecar exists.
        metadata_suffix (str): Suffix appended to the video stem for the cache file
            (via Path.with_suffix). Default ".metadata.json".

    Returns:
        dict with keys:
            "n_frames" (int): Total frame count.
            "frame_size" (tuple[int, int]): (height, width).
            "fps" (float | None): Frames per second, or None if unavailable.
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
