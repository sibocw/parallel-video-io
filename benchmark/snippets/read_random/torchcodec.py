# Read specific frames by index.
from torchcodec.decoders import VideoDecoder


def get_frames(path, indices):
    decoder = VideoDecoder(path, seek_mode="exact")
    return decoder.get_frames_at(indices).data
