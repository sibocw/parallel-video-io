# Write NumPy frames to an MP4.
from pvio.io import write_frames_to_video


def save(frames, fps, path):
    write_frames_to_video(path, list(frames), fps=fps)
