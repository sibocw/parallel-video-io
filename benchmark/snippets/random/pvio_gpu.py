# Precise random-access decode on the GPU.
from pvio.video import EncodedVideo


def get_frames(path, indices):
    video = EncodedVideo(path, device="cuda")
    video.setup()
    return [video.read_frame(i) for i in indices]
