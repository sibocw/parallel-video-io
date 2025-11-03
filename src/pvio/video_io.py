import numpy as np
import json
import logging
import av
import imageio.v2 as imageio
from pathlib import Path
from fractions import Fraction


class _InputVideo:
    """
    Strict CFR, zero-start, single-stream (video-only), integer-PTS video helper.

    - Opens with av.open(path) internally.
    - Validates: exactly one stream; it must be a video stream.
    - Rejects non-zero container/stream start times.
    - Rejects VFR: requires integer ticks_per_frame.
    - Provides exact integer mapping frame <-> PTS.
    - Private API for frame reading; public wrapper is read_frames_from_video().
    """

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self.container = av.open(self.video_path.as_posix())
        self._closed = False

        # ---- Stream checks: exactly one stream and it must be video ----
        if len(self.container.streams) != 1:
            raise ValueError(
                f"Expected exactly one stream; found {len(self.container.streams)}."
            )
        stream = self.container.streams[0]
        if stream.type != "video":
            raise ValueError("Single stream must be a video stream.")

        # ---- Container start time (AV_TIME_BASE units) ----
        ct = self.container.start_time
        if ct not in (None, 0):
            raise ValueError(
                f"Non-zero container.start_time={ct} (AV_TIME_BASE units). Refusing."
            )

        # ---- Stream timing sanity ----
        if stream.average_rate is None:
            raise ValueError(
                "Video stream has no average_rate (FPS). Refusing VFR/unknown rate."
            )
        if stream.time_base is None:
            raise ValueError(
                "Video stream has no time_base. Cannot do precise integer seeking."
            )

        self.stream = stream
        self.fps: Fraction = Fraction(stream.average_rate)
        self.time_base: Fraction = Fraction(stream.time_base)

        if self.fps.numerator <= 0 or self.fps.denominator <= 0:
            raise ValueError(f"Invalid FPS {self.fps}.")

        st = stream.start_time  # in stream time_base ticks, or None
        if st not in (None, 0):
            raise ValueError(f"Non-zero stream.start_time={st} ticks. Refusing.")

        # Require integer ticks per frame
        ticks_per_frame = Fraction(1, 1) / (self.time_base * self.fps)
        if ticks_per_frame.denominator != 1:
            raise ValueError(
                f"VFR or incompatible rate/time_base: "
                f"time_base={self.time_base}, fps={self.fps} "
                f"-> ticks_per_frame={ticks_per_frame}, which is not an integer."
            )
        self.ticks_per_frame: int = int(ticks_per_frame)

        self.n_frames = check_num_frames(self.video_path)

    def close(self):
        if not self._closed:
            self.container.close()
            self._closed = True

    def __enter__(self) -> "_InputVideo":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def frame_idx_to_pts(self, frame_idx: int) -> int:
        if frame_idx < 0:
            raise IndexError("Negative frame index.")
        return frame_idx * self.ticks_per_frame

    def pts_to_frame_idx(self, pts: int) -> int:
        if pts < 0:
            raise IndexError("PTS before stream start.")
        q, r = divmod(pts, self.ticks_per_frame)
        if r != 0:
            raise ValueError(
                f"PTS {pts} not aligned to frame grid (ticks_per_frame={self.ticks_per_frame})."
            )
        return q

    def get_fps(self, as_float: bool = True) -> float | Fraction:
        if as_float:
            return float(self.fps)
        else:
            return self.fps

    def _exact_seek_and_read(self, frame_idx: int) -> av.video.frame.VideoFrame:
        """
        Seek to the exact frame using integer PTS only.
        Raises if timestamps overshoot or are missing.
        """
        target_pts = self.frame_idx_to_pts(frame_idx)
        self.container.seek(
            target_pts, any_frame=False, backward=True, stream=self.stream
        )

        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                raise ValueError("Decoded frame without PTS. Refusing.")
            curr_idx = self.pts_to_frame_idx(int(frame.pts))
            logging.debug(f"!!! Decoded frame at index {curr_idx}.")

            if curr_idx < frame_idx:
                continue
            elif curr_idx == frame_idx:
                return frame
            else:
                raise IndexError(
                    f"Overshot target PTS (got {curr_idx}, wanted {frame_idx}). "
                    "Stream likely VFR or has non-conforming timestamps."
                )

        raise IndexError(f"Target frame {frame_idx} not found before EOF.")

    def read_frames(
        self,
        frame_indices: list[int] | None,
        max_sequential_decodes: int,
    ) -> list[np.ndarray]:
        logger = logging.getLogger(__name__)

        # Simplest case: read all frames
        if frame_indices is None:
            images: list[np.ndarray] = []
            for frame in self.container.decode(self.stream):
                if frame.pts is None:
                    raise ValueError("Decoded frame without PTS. Refusing.")
                # Enforce grid alignment (check pts validity and raise error if needed)
                _ = self.pts_to_frame_idx(int(frame.pts))  # just for validation
                images.append(frame.to_ndarray(format="rgb24"))
            return images

        # Empty list case
        if len(frame_indices) == 0:
            logging.getLogger(__name__).warning(
                "Requested zero frames from video. Returning empty list."
            )
            return []

        # Deduplicate & sort
        frame_indices_sorted = sorted(set(frame_indices))
        if frame_indices_sorted[0] < 0:
            raise IndexError("Negative frame index requested.")
        if frame_indices_sorted[-1] >= self.n_frames:
            raise IndexError(
                f"Requested frame index {frame_indices_sorted[-1]} "
                f"beyond end of video (n_frames={self.n_frames})."
            )

        images_by_idx: dict[int, np.ndarray] = {}

        # Prime at first requested index
        first_idx = frame_indices_sorted[0]
        frame = self._exact_seek_and_read(first_idx)
        images_by_idx[first_idx] = frame.to_ndarray(format="rgb24")
        last_idx_obtained = first_idx

        # Process remaining indices
        for target_idx in frame_indices_sorted[1:]:
            logger.debug(
                f"Reading target frame {target_idx} (last obtained: {last_idx_obtained})."
            )
            gap = target_idx - last_idx_obtained
            if gap <= 0:
                raise RuntimeError(
                    f"Internal logic error: non-increasing frame request sequence."
                    f" last={last_idx_obtained}, next={target_idx}."
                )

            n_decodes_needed = gap - 1
            logger.debug(
                f"Gap of {gap} frames; need {n_decodes_needed} decodes in between."
            )
            if n_decodes_needed >= max_sequential_decodes:
                # Too far apart: seek directly
                logger.debug(f"Gap too large; seeking directly to frame {target_idx}.")
                frame = self._exact_seek_and_read(target_idx)
                images_by_idx[target_idx] = frame.to_ndarray(format="rgb24")
                last_idx_obtained = target_idx
            else:
                # Strict sequential decode
                for frame in self.container.decode(self.stream):
                    if frame.pts is None:
                        raise ValueError("Decoded frame without PTS. Refusing.")
                    curr_idx = self.pts_to_frame_idx(int(frame.pts))
                    logging.debug(f"!!! Decoded frame at index {curr_idx}.")
                    logging.debug(f"total n frames according to container: {self.container.streams[0].frames}")

                    if curr_idx == target_idx:
                        images_by_idx[curr_idx] = frame.to_ndarray(format="rgb24")
                        last_idx_obtained = curr_idx
                        break

        # Return in caller's requested order
        try:
            ordered = [images_by_idx[i] for i in frame_indices]
        except KeyError as e:
            raise IndexError(f"Requested frame {e.args[0]} was not decoded.") from None

        return ordered


def read_frames_from_video(
    video_path: Path | str,
    frame_indices: list[int] | None = None,
    max_sequential_decodes: int = 75,
    return_fps: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], float]:
    """Smartly read specific frames from a video file.

    Args:
        video_path (Path | str): Path to the video file.
        frame_indices (list[int] | None): List of frame indices to read.
            If None, read all frames.
        max_sequential_decodes (int): In case the requested frames are
            not contiguous, we will sort the frame indices and decode in
            increasing order. If the requested indices are close enough,
            we decode frames sequentially to avoid repeated seeking. If the
            next requested frame index is far apart, we perform a precise
            seek to the next desired frame. This parameter sets the
            threshold for deciding whether to decode sequentially or seek
            directly. Default is 75.
        return_fps (bool): If True, also return the video's FPS. Default is False.

    Returns:
        frames (list[np.ndarray]): List of frames as numpy arrays.
        (Only if return_fps is True) fps (float): FPS of the video.
    """
    with _InputVideo(video_path) as vid:
        frames = vid.read_frames(frame_indices, max_sequential_decodes)
        if return_fps:
            return frames, vid.get_fps(as_float=True)
        else:
            return frames


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
    log_level: int = logging.INFO,
):
    """Write a sequence of frames to a video file.

    Args:
        video_path (Path | str): Path to save the video file.
        frames (list[np.ndarray]): List of frames as numpy arrays (in
            [height, width, channels] format).
        fps (float): Frames per second for the output video.
        codec (str): Codec to use. Default: 'libx264'.
        ffmpeg_params (list[str]): Additional ffmpeg parameters.
            Default is a set of parameters for high-quality H.264 encoding.
            (see _default_ffmpeg_params_for_video_writing).
        log_interval (int | None): If set, log progress every
            `log_interval` frames at the specified log level.
        log_level (int): Logging level for progress. Default: logging.INFO.
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
                logging.log(log_level, f"Written frame {i + 1}/{len(frames)}")


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
):
    """Get number of frames, frame size, and FPS of a video file.

    Args:
        video_path (Path | str): Path to the video file.
        cache_metadata (bool): Whether to cache the metadata to a JSON
            file. Default is True.
        use_cached_metadata (bool): Whether to use cached metadata if
            available. Default is True.
        metadata_suffix (str): Suffix to use for the metadata cache file.
            Default is ".metadata.json".

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
            print(f"Corrupted metadata cache file {cache_path}")
            raise e
    else:
        n_frames = check_num_frames(video_path)
        sample_frames, fps = read_frames_from_video(
            video_path, frame_indices=[0], return_fps=True
        )
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
