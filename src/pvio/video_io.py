import numpy as np
import json
import logging
import imageio.v2 as imageio
from pathlib import Path


def read_frames_from_video(
    video_path: Path | str, frame_indices: list[int] | None = None
) -> tuple[list[np.ndarray], float]:
    """Read specific frames from a video file.

    Args:
        video_path (Path | str): Path to the video file.
        frame_indices (list[int] | None): List of frame indices to read. If None, read
            all frames.

    Raises:
        ValueError: If the video file cannot be read.
        IndexError: If the frame indices are invalid.

    Returns:
        frames (list[np.ndarray]): List of frames as numpy arrays.
        fps (float): FPS of the video.
    """
    frames = []
    with imageio.get_reader(video_path) as reader:
        if frame_indices is None:
            frame_indices = list(range(reader.count_frames()))
        for idx in frame_indices:
            frames.append(reader.get_data(idx))
        fps = reader.get_meta_data().get("fps", None)
    return frames, fps


_default_ffmpeg_params_for_video_writing = [
    "-crf",
    "15",  # Lower CRF = higher quality (15 is very high quality)
    "-preset",
    "slow",  # Slower preset = better compression efficiency
    "-profile:v",
    "high",  # Use high profile for better compression
    "-level",
    "4.0",  # H.264 level
]


def write_frames_to_video(
    video_path: Path | str,
    frames: list[np.ndarray],
    fps: float,
    codec: str = "libx264",
    ffmpeg_params: list[str] = _default_ffmpeg_params_for_video_writing,
    log_interval: int | None = None,
    logger: logging.Logger | None = None,
):
    """Write a sequence of frames to a video file.

    Args:
        video_path (Path | str): Path to save the video file.
        frames (list[np.ndarray]): List of frames as numpy arrays (in
            [height, width, channels] format).
        fps (float): Frames per second for the output video.
        codec (str): Codec to use. Default: 'libx264'.
        ffmpeg_params (list[str]): Additional ffmpeg parameters. Default is a set of
            parameters for high-quality H.264 encoding (see
            _default_ffmpeg_params_for_video_writing).
        log_interval (int | None): If set, log progress every `log_interval` frames
            using the specified logger.
        logger (logging.Logger | None): Logger to use for progress logging. If None, use
            the logger from `__main__`.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

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
    logger: logging.Logger | None = None,
):
    """Get number of frames, frame size, and FPS of a video file.

    Args:
        video_path (Path | str): Path to the video file.
        cache_metadata (bool): Whether to cache the metadata to a JSON file. Default is
            True.
        use_cached_metadata (bool): Whether to use cached metadata if available. Default
            is True.
        metadata_suffix (str): Suffix to use for the metadata cache file. Default is
            ".metadata.json".
        logger (logging.Logger | None): Logger to use for logging. If None, use the
            logger from `__main__`.

    Returns:
        dict: A dictionary containing the video metadata.
    """
    metadata = {}

    video_path = Path(video_path)
    cache_path = video_path.with_suffix(metadata_suffix)
    if use_cached_metadata and cache_path.is_file():
        try:
            with open(cache_path, "r") as f:
                metadata = json.load(f)
            n_frames = metadata["n_frames"]
            frame_size = tuple(metadata["frame_size"])
            fps = metadata["fps"]
        except Exception as e:
            logger.critical(f"Corrupted metadata cache file {cache_path}")
            raise e
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
            with open(cache_path, "w") as f:
                json.dump(metadata, f, indent=2)

    return {"n_frames": n_frames, "frame_size": frame_size, "fps": fps}
