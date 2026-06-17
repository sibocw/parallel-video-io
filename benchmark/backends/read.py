"""Frame-reading backends for the sequential and random-access benchmarks.

Every backend exposes:

* ``available()`` -> (ok, reason)
* ``sequential(path, n_frames)`` -> number of frames decoded (streams one at a
  time; must not accumulate, so 4K videos don't blow up memory).
* ``random(path, indices)`` -> ``(k, H, W, 3)`` uint8 ndarray on CPU, used for
  both timing and seek-correctness checking.

These mirror the *minimal idiomatic* usage of each library; the matching
user-code snippets in ``benchmark/snippets/read`` are kept in sync for the
lines-of-code metric.
"""

from __future__ import annotations

import numpy as np
import torch


class ReadBackend:
    name: str = "base"
    device: str = "cpu"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def sequential(self, path: str, n_frames: int) -> int:
        raise NotImplementedError

    def random(self, path: str, indices: list[int]) -> np.ndarray:
        raise NotImplementedError


def _chw01_to_hwc_u8(frame: torch.Tensor) -> np.ndarray:
    arr = (frame.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return arr.permute(1, 2, 0).cpu().numpy()


# --------------------------------------------------------------------------- #
# PVIO
# --------------------------------------------------------------------------- #
class PvioEncoded(ReadBackend):
    """PVIO's EncodedVideo (TorchCodec backend with forward buffering) — the
    same code path the parallel DataLoader uses."""

    name = "pvio"
    device = "cpu"

    def sequential(self, path: str, n_frames: int) -> int:
        from pvio.video import EncodedVideo

        vid = EncodedVideo(path, buffer_size=64)
        vid.setup()
        count = 0
        for i in range(len(vid)):
            vid.read_frame(i)
            count += 1
        vid.close()
        return count

    def random(self, path: str, indices: list[int]) -> np.ndarray:
        from pvio.video import EncodedVideo

        # buffer_size=1 makes each read a single true seek (fair per-frame timing).
        vid = EncodedVideo(path, buffer_size=1)
        vid.setup()
        frames = [_chw01_to_hwc_u8(vid.read_frame(i)) for i in indices]
        vid.close()
        return np.stack(frames)


class PvioSimple(ReadBackend):
    """PVIO's convenience reader (imageio/FFmpeg), ``read_frames_from_video``."""

    name = "pvio_simple"
    device = "cpu"

    def sequential(self, path: str, n_frames: int) -> int:
        import imageio.v2 as imageio

        # Mirror read_frames_from_video but stream instead of accumulating.
        count = 0
        with imageio.get_reader(path) as reader:
            for _ in reader:
                count += 1
        return count

    def random(self, path: str, indices: list[int]) -> np.ndarray:
        from pvio.io import read_frames_from_video

        frames, _ = read_frames_from_video(path, frame_indices=indices)
        return np.stack(frames)


# --------------------------------------------------------------------------- #
# TorchCodec (raw) — CPU and CUDA/NVDEC
# --------------------------------------------------------------------------- #
class _TorchCodec(ReadBackend):
    def sequential(self, path: str, n_frames: int) -> int:
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(path, seek_mode="approximate", device=self.device)
        count = 0
        for _ in dec:
            count += 1
        if self.device == "cuda":
            torch.cuda.synchronize()
        return count

    def random(self, path: str, indices: list[int]) -> np.ndarray:
        from torchcodec.decoders import VideoDecoder

        dec = VideoDecoder(path, seek_mode="exact", device=self.device)
        batch = dec.get_frames_at(indices).data  # NCHW uint8
        return batch.permute(0, 2, 3, 1).cpu().numpy()


class TorchCodecCPU(_TorchCodec):
    name = "torchcodec_cpu"
    device = "cpu"


class TorchCodecCUDA(_TorchCodec):
    name = "torchcodec_cuda"
    device = "cuda"

    def available(self) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        return True, ""


# --------------------------------------------------------------------------- #
# Decord (CPU only in the PyPI wheel)
# --------------------------------------------------------------------------- #
class DecordCPU(ReadBackend):
    name = "decord_cpu"
    device = "cpu"

    def available(self) -> tuple[bool, str]:
        try:
            import decord  # noqa: F401
        except Exception as e:  # pragma: no cover
            return False, f"import failed: {e}"
        return True, ""

    def sequential(self, path: str, n_frames: int) -> int:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        count = 0
        for i in range(len(vr)):
            _ = vr[i]
            count += 1
        return count

    def random(self, path: str, indices: list[int]) -> np.ndarray:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        return vr.get_batch(indices).asnumpy()


# --------------------------------------------------------------------------- #
# PyAV (FFmpeg bindings)
# --------------------------------------------------------------------------- #
class PyAVCPU(ReadBackend):
    name = "pyav_cpu"
    device = "cpu"

    def available(self) -> tuple[bool, str]:
        try:
            import av  # noqa: F401
        except Exception as e:  # pragma: no cover
            return False, f"import failed: {e}"
        return True, ""

    def sequential(self, path: str, n_frames: int) -> int:
        import av

        count = 0
        with av.open(path) as container:
            for _ in container.decode(video=0):
                count += 1
        return count

    def random(self, path: str, indices: list[int]) -> np.ndarray:
        import av

        out = []
        with av.open(path) as container:
            stream = container.streams.video[0]
            avg_rate = stream.average_rate
            time_base = stream.time_base
            for idx in indices:
                # Seek to the nearest keyframe before the target, then decode forward.
                target_pts = int(idx / avg_rate / time_base)
                container.seek(target_pts, stream=stream, backward=True)
                chosen = None
                for frame in container.decode(stream):
                    frame_idx = int(round(float(frame.pts * time_base * avg_rate)))
                    if frame_idx >= idx:
                        chosen = frame
                        break
                if chosen is None:
                    raise RuntimeError(f"PyAV could not decode frame {idx}")
                out.append(chosen.to_ndarray(format="rgb24"))
        return np.stack(out)


# --------------------------------------------------------------------------- #
# OpenCV
# --------------------------------------------------------------------------- #
class OpenCVCPU(ReadBackend):
    name = "opencv_cpu"
    device = "cpu"

    def available(self) -> tuple[bool, str]:
        try:
            import cv2  # noqa: F401
        except Exception as e:  # pragma: no cover
            return False, f"import failed: {e}"
        return True, ""

    def sequential(self, path: str, n_frames: int) -> int:
        import cv2

        cap = cv2.VideoCapture(path)
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        return count

    def random(self, path: str, indices: list[int]) -> np.ndarray:
        import cv2

        cap = cv2.VideoCapture(path)
        out = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"OpenCV could not read frame {idx}")
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(out)


READ_BACKENDS: list[ReadBackend] = [
    PvioEncoded(),
    PvioSimple(),
    TorchCodecCPU(),
    TorchCodecCUDA(),
    DecordCPU(),
    PyAVCPU(),
    OpenCVCPU(),
]
