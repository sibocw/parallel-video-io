# Merge frames into an MP4 on the GPU (NVENC).
from pvio.io import write_frames_to_video


def save(frames, fps, path):
    write_frames_to_video(path, list(frames), fps=fps, mode="gpu")
