# Sequential decode.
from torchcodec.decoders import VideoDecoder


def decode_all(path):
    return list(VideoDecoder(path))
