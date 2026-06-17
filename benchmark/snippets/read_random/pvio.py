# Read specific frames by index.
from pvio.io import read_frames_from_video


def get_frames(path, indices):
    frames, _ = read_frames_from_video(path, frame_indices=indices)
    return frames
