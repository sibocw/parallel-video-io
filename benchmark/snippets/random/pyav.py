# Random-access decode with PyAV (seek to keyframe, decode forward).
import av


def get_frames(path, indices):
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        rate, tb = stream.average_rate, stream.time_base
        for idx in indices:
            container.seek(int(idx / rate / tb), stream=stream, backward=True)
            for frame in container.decode(stream):
                if round(float(frame.pts * tb * rate)) >= idx:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    break
    return frames
