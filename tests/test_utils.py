"""Shared utilities for pvio tests."""

import numpy as np
import torch
from typing import Callable

from pvio.video import Video
from pvio.torch_tools import VideoCollectionDataset


def make_frames_with_stride(n_frames: int, stride: int = 10, h: int = 32, w: int = 32):
    """Create frames with values spaced by stride to handle lossy compression.

    Each frame has a unique value: 0, 10, 20, 30, ... (up to 250 max).
    This spacing helps distinguish frames even after video compression.

    Args:
        n_frames (int): Number of frames to create.
        stride (int): Value spacing between frames. Default: 10.
        h (int): Frame height. Default: 32.
        w (int): Frame width. Default: 32.

    Returns:
        list[np.ndarray]: List of frames as uint8 numpy arrays.
    """
    frames = []
    for i in range(n_frames):
        value = (i * stride) % 256
        frame = np.full((h, w, 3), fill_value=value, dtype=np.uint8)
        frames.append(frame)
    return frames


def make_simple_frames(n: int, h: int = 32, w: int = 32, channels: int = 3):
    """Create simple frames for basic io tests.

    Varies the red channel to help identify frames.

    Args:
        n (int): Number of frames to create.
        h (int): Frame height. Default: 32.
        w (int): Frame width. Default: 32.
        channels (int): Number of channels. Default: 3.

    Returns:
        list[np.ndarray]: List of frames as uint8 numpy arrays.
    """
    frames = []
    for i in range(n):
        arr = np.zeros((h, w, channels), dtype=np.uint8)
        arr[..., 0] = i  # Vary red channel to identify frames
        frames.append(arr)
    return frames


class DummyVideo(Video):
    """Minimal dummy video for testing without real files."""

    def __init__(self, n_frames: int = 3, frame_range: tuple[int, int] | None = None):
        # Use a fake path for testing
        super().__init__(path="/fake/path/test.mp4", frame_range=frame_range)
        self.n_frames = n_frames

    def _validate_init_params(self) -> None:
        """Dummy validation - always passes."""
        pass

    def _load_metadata(self) -> tuple[int, tuple[int, int], float]:
        """Return dummy metadata."""
        return self.n_frames, (4, 5), 30.0  # height, width, fps

    def _read_frame(
        self, index: int, transform: Callable | None = None
    ) -> torch.Tensor:
        """Generate a dummy frame with pattern based on index."""
        frame = torch.ones(3, 4, 5) * index  # CHW format
        if transform is not None:
            frame = transform(frame)
        return frame


class DummyDataset(VideoCollectionDataset):
    """Lightweight dataset for testing without real videos."""

    def __init__(self, videos=None):
        if videos is None:
            videos = [DummyVideo()]
        super().__init__(videos)

    def assign_workers(self, n_loading_workers: int, min_frames_per_worker: int = None):
        self.worker_assignments = [[] for _ in range(n_loading_workers)]

    def __iter__(self):
        for i in range(3):
            yield {
                "frame": torch.ones(3, 4, 5) * i,
                "video_id": i,
                "frame_id": i,
            }
