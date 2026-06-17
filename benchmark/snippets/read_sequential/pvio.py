# Decode every frame of a video.
from pvio.io import read_frames_from_video


def decode_all(path):
    frames, _ = read_frames_from_video(path)
    return frames
