"""Shared utilities for pvio tests."""

import numpy as np


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
