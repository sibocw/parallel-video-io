"""Encoding backends: merge a sequence of NumPy frames into an H.264 MP4.

Every backend exposes ``available()`` and ``encode(frames, fps, out_path,
quality)``. ``quality`` is on the 0-51 H.264 quantiser scale (CRF for libx264,
QP for NVENC); backends without a quality knob (OpenCV's mp4v) set
``tunable = False`` and ignore it. These mirror the minimal idiomatic usage of
each library.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np


class EncodeBackend:
    name: str = "base"
    device: str = "cpu"
    tunable: bool = True  # has a quality knob worth sweeping for the Pareto front

    def available(self) -> tuple[bool, str]:
        return True, ""

    def quality_display(self, q: int) -> int:
        """Quality value to record in results (override if the backend remaps it)."""
        return q

    def encode(self, frames: np.ndarray, fps: int, out_path: str, quality: int) -> None:
        """Encode ``frames`` (N, H, W, 3) uint8 RGB to ``out_path``."""
        raise NotImplementedError


class PvioCPUEncode(EncodeBackend):
    """PVIO write_frames_to_video, CPU path (libx264, CRF)."""

    name = "pvio_cpu"
    device = "cpu"

    def encode(self, frames, fps, out_path, quality) -> None:
        from pvio.io import write_frames_to_video

        write_frames_to_video(
            out_path, list(frames), fps=fps, mode="cpu", quality=quality, quiet=True
        )


class PvioGPUEncode(EncodeBackend):
    """PVIO write_frames_to_video, GPU path (H.264 NVENC, constant QP)."""

    name = "pvio_gpu"
    device = "cuda"

    def available(self) -> tuple[bool, str]:
        try:
            from pvio import _accel
        except Exception as e:  # pragma: no cover
            return False, f"import failed: {e}"
        if not _accel.cuda_available():
            return False, "no CUDA device"
        if _accel.nvenc_ffmpeg_exe() is None:
            return False, "no NVENC-capable ffmpeg"
        return True, ""

    def encode(self, frames, fps, out_path, quality) -> None:
        from pvio.io import write_frames_to_video

        write_frames_to_video(
            out_path, list(frames), fps=fps, mode="gpu", quality=quality, quiet=True
        )


class PyAVEncode(EncodeBackend):
    """PyAV (FFmpeg bindings), libx264 at matched CRF/preset."""

    name = "pyav"
    device = "cpu"

    def available(self) -> tuple[bool, str]:
        try:
            import av  # noqa: F401
        except Exception as e:  # pragma: no cover
            return False, f"import failed: {e}"
        return True, ""

    def encode(self, frames, fps, out_path, quality) -> None:
        import av

        n, h, w, _ = frames.shape
        with av.open(out_path, "w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(quality), "preset": "slow"}
            for i in range(n):
                frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():  # flush
                container.mux(packet)


class OpenCVEncode(EncodeBackend):
    """OpenCV VideoWriter (MJPEG). Quality is mapped from the CRF scale to JPEG %."""

    name = "opencv"
    device = "cpu"
    tunable = True

    @staticmethod
    def _mjpeg_q(crf: int) -> int:
        """Map CRF 0-51 (lower=better) to JPEG quality 0-100 (higher=better)."""
        return max(0, min(100, round(100 * (1 - crf / 51))))

    def quality_display(self, q: int) -> int:
        return self._mjpeg_q(q)

    def available(self) -> tuple[bool, str]:
        try:
            import cv2  # noqa: F401
        except Exception as e:  # pragma: no cover
            return False, f"import failed: {e}"
        return True, ""

    def encode(self, frames, fps, out_path, quality) -> None:
        import cv2

        n, h, w, _ = frames.shape
        mjpeg_q = self._mjpeg_q(quality)
        # MJPEG requires AVI container; write to a sidecar then move.
        avi_path = str(Path(out_path).with_suffix(".avi"))
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, mjpeg_q)
        for i in range(n):
            writer.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        writer.release()
        shutil.move(avi_path, out_path)


ENCODE_BACKENDS: list[EncodeBackend] = [
    PvioCPUEncode(),
    PvioGPUEncode(),
    PyAVEncode(),
    OpenCVEncode(),
]
