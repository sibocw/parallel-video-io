# Merge frames into an MP4 with PyAV (libx264).
import av


def save(frames, fps, path):
    n, h, w, _ = frames.shape
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "slow"}
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
