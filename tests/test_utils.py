"""Shared utilities for pvio tests."""

import numpy as np
import torch

from pvio.torch import VideoCollectionDataset, Video


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
    """Create simple frames for basic video_io tests.

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

    def __init__(self, n_frames: int = 3):
        self.n_frames = n_frames
        self._Video__setup_done = False

    def setup(self):
        if self._Video__setup_done:
            return
        self._Video__setup_done = True

    def __len__(self):
        return self.n_frames

    def read_frame(self, index: int, transform=None):
        frame = torch.ones(3, 4, 5) * index
        if transform is not None:
            frame = transform(frame)
        return {"frame": frame, "frame_idx": index}

    def _load_metadata(self):
        return self.n_frames, (3, 4, 5), 30.0

    def _post_setup(self):
        pass


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
                "video_idx": i,
                "frame_idx": i,
            }
