# Precise random-access decode with Decord (frame-accurate seek).
import decord


def get_frames(path, indices):
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    frames = []
    for i in indices:
        vr.seek_accurate(i)
        frames.append(vr.next().asnumpy())
    return frames
