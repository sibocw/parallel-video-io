# Decode every frame of a video.
import av


def decode_all(path):
    with av.open(path) as container:
        return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
