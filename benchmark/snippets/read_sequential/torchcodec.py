# Decode every frame of a video.
from torchcodec.decoders import VideoDecoder


def decode_all(path):
    return list(VideoDecoder(path))
