# Precise random-access decode on the CPU.
from pvio.video import EncodedVideo


def get_frames(path, indices):
    video = EncodedVideo(path, device="cpu")
    video.setup()
    return [video.read_frame(i) for i in indices]
