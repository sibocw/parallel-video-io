# Read specific frames by index.
import decord


def get_frames(path, indices):
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    return vr.get_batch(indices).asnumpy()
