# Sequential decode on the CPU.
from pvio.video import EncodedVideo


def decode_all(path):
    video = EncodedVideo(path, device="cpu")
    video.setup()
    return [video.read_frame(i) for i in range(len(video))]
