"""Frame-writing backends: NumPy frames -> H.264 MP4.

Each backend writes the same uint8 RGB frames to a file. The benchmark then
measures encode throughput, output file size, and reconstruction quality
(PSNR/SSIM vs the source) so the speed/size/quality trade-off is visible.

PVIO and PyAV are driven at the same CRF for a fair comparison. OpenCV's
VideoWriter does not expose CRF (rate control is fixed by the FOURCC codec),
so its quality/size point is reported as-is and flagged.
"""

from __future__ import annotations

import numpy as np


class WriteBackend:
    name = "base"
    crf_controlled = True

    def available(self) -> tuple[bool, str]:
        return True, ""

    def write(self, frames: np.ndarray, fps: int, out_path: str, crf: int) -> None:
        """Encode ``frames`` (N, H, W, 3) uint8 RGB to ``out_path``."""
        raise NotImplementedError


class PvioWriter(WriteBackend):
    """PVIO's write_frames_to_video (imageio/FFmpeg, libx264)."""

    name = "pvio"

    def write(self, frames, fps, out_path, crf) -> None:
        from pvio.io import write_frames_to_video

        # imageio already sets -pix_fmt yuv420p for libx264; don't duplicate it.
        ffmpeg_params = ["-crf", str(crf), "-preset", "medium"]
        write_frames_to_video(
            out_path, list(frames), fps=fps, ffmpeg_params=ffmpeg_params
        )


class PvioNvencWriter(WriteBackend):
    """PVIO's auto path on a GPU machine: H.264 NVENC (GPU encode).

    Uses ``write_frames_to_video`` with no explicit codec, which is exactly what
    a user gets by default on a CUDA machine with an NVENC-capable FFmpeg. NVENC
    uses constant-QP rate control (visually lossless, ~CRF 20), so the ``crf``
    argument does not apply and is ignored.
    """

    name = "pvio_nvenc"
    crf_controlled = False

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

    def write(self, frames, fps, out_path, crf) -> None:
        from pvio.io import write_frames_to_video

        write_frames_to_video(out_path, list(frames), fps=fps)


class PyAVWriter(WriteBackend):
    name = "pyav"

    def available(self) -> tuple[bool, str]:
        try:
            import av  # noqa: F401
        except Exception as e:  # pragma: no cover
            return False, f"import failed: {e}"
        return True, ""

    def write(self, frames, fps, out_path, crf) -> None:
        import av

        n, h, w, _ = frames.shape
        with av.open(out_path, "w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf), "preset": "medium"}
            for i in range(n):
                frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():  # flush
                container.mux(packet)


class OpenCVWriter(WriteBackend):
    name = "opencv"
    crf_controlled = False

    def available(self) -> tuple[bool, str]:
        try:
            import cv2  # noqa: F401
        except Exception as e:  # pragma: no cover
            return False, f"import failed: {e}"
        return True, ""

    def write(self, frames, fps, out_path, crf) -> None:
        import cv2

        n, h, w, _ = frames.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open")
        for i in range(n):
            writer.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        writer.release()


WRITE_BACKENDS: list[WriteBackend] = [
    PvioWriter(),
    PvioNvencWriter(),
    PyAVWriter(),
    OpenCVWriter(),
]
