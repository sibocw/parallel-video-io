# Sequential decode on the GPU.
from pvio.video import EncodedVideo


def decode_all(path):
    video = EncodedVideo(path, device="cuda")
    video.setup()
    return [video.read_frame(i) for i in range(len(video))]
