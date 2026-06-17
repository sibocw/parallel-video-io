# Sequential decode with Decord.
import decord


def decode_all(path):
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    return [vr[i] for i in range(len(vr))]
