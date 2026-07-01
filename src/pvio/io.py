import numpy as np
import json
import logging
import os
import sys
import tempfile
import contextlib
import av
import imageio.v2 as imageio
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm

from . import _accel


logger = logging.getLogger(__name__)


class VideoMetadata(NamedTuple):
    """Basic video metadata returned by :func:`get_video_metadata`.

    A lightweight typed record (tuple-compatible) with three fields:

    Attributes:
        n_frames: Total frame count.
        frame_size: ``(height, width)`` in pixels.
        fps: Frames per second, or ``None`` if unavailable (e.g. image
            directories).
    """

    n_frames: int
    frame_size: tuple[int, int]
    fps: float | None


@contextlib.contextmanager
def _imageio_ffmpeg_exe(exe: str | None):
    """Temporarily point imageio-ffmpeg at *exe* via ``IMAGEIO_FFMPEG_EXE``.

    Used to route NVENC encodes through a system FFmpeg, since the binary
    bundled with imageio-ffmpeg is usually built without NVENC. A no-op when
    *exe* is None. Restores the previous environment on exit.
    """
    if exe is None:
        yield
        return
    sentinel = object()
    prev = os.environ.get("IMAGEIO_FFMPEG_EXE", sentinel)
    os.environ["IMAGEIO_FFMPEG_EXE"] = exe
    try:
        yield
    finally:
        if prev is sentinel:
            os.environ.pop("IMAGEIO_FFMPEG_EXE", None)
        else:
            os.environ["IMAGEIO_FFMPEG_EXE"] = prev


def _select_pixel_format(fmt) -> str:
    """Pick a ``to_ndarray`` target that preserves *fmt*'s depth and channels.

    *fmt* is a PyAV :class:`av.video.format.VideoFormat`. The chosen target keeps
    the source's native bit depth (uint16 for >8-bit sources, uint8 otherwise)
    and channel layout, rather than collapsing everything to 8-bit RGB the way
    imageio's reader does. This matters for lossless high-bit-depth videos (e.g.
    FFV1 ``gbrap16le``): an 8-bit decode keeps only the high byte, so small
    16-bit values (in the hundreds) all map to 0/1/2 and the real data is lost.

    A source alpha channel is preserved (RGBA output); a single-plane grayscale
    source decodes to a 2-D ``(H, W)`` array. Everything else decodes to RGB.
    """
    high = max(c.bits for c in fmt.components) > 8
    has_alpha = any(c.is_alpha for c in fmt.components)
    n_color = sum(1 for c in fmt.components if not c.is_alpha)
    is_gray = n_color == 1 and not fmt.is_rgb
    if has_alpha:
        return "rgba64le" if high else "rgba"
    if is_gray:
        return "gray16le" if high else "gray"
    return "rgb48le" if high else "rgb24"


def _index_at(time_base, rate, ts) -> int:
    """Convert a stream timestamp *ts* to a (rounded) frame index."""
    return int(round(float(ts * time_base * rate)))


def _seek_landing(container, stream, index: int, rate, time_base) -> int | None:
    """Seek to *index* and report which frame the seek actually lands on.

    Demuxes a single packet (no decoding) and maps its timestamp back to a frame
    index. Returns ``None`` if the packet has no timestamp. A seek lands on the
    nearest seek point at or before the target; how far before is exactly what
    decides the read strategy in :func:`_read_indexed_frames`.
    """
    container.seek(int((index / rate) / time_base), stream=stream)
    for packet in container.demux(stream):
        if packet.pts is None:
            return None
        return _index_at(time_base, rate, packet.pts)
    return None


def _seek_granularity(container, stream, probe_index: int, rate, time_base) -> int | None:
    """Estimate the spacing (in frames) between seek points near *probe_index*.

    Measured empirically because no metadata is reliable: an FFV1/MKV stream
    exposes seek points roughly every dozen frames (container cues) while
    reporting only one keyframe, whereas H.264 only lets a seek land on a GOP
    keyframe hundreds of frames apart. Two cheap demux-only probes locate two
    adjacent seek points; their distance is the granularity ``g``.

    ``g`` drives the per-frame-seek vs forward-decode choice: jumping more than
    ``g`` frames ahead is cheaper via a fresh seek (it skips to a nearer seek
    point), while a shorter hop is cheaper by decoding forward (a re-seek would
    just land back at the same seek point and re-decode). Returns ``None`` if
    timestamps are unavailable.
    """
    k1 = _seek_landing(container, stream, probe_index, rate, time_base)
    if k1 is None:
        return None
    if k1 <= 0:
        # The probe already lands at the start; treat seeks as fine-grained.
        return 1
    k0 = _seek_landing(container, stream, k1 - 1, rate, time_base)
    if k0 is None:
        return None
    return max(1, k1 - k0)


def _read_indexed_frames(
    container, stream, frame_indices: list[int], rate, time_base
) -> list[np.ndarray] | None:
    """Read specific *frame_indices* efficiently for any codec.

    Walks the requested frames in sorted order with a single decoder, keeping it
    running across frames. For each next frame it either keeps decoding forward
    (when the frame is within one seek-granularity ``g`` of the current position)
    or seeks afresh (when it is further ahead, or behind). This is optimal at both
    extremes: sparse far-apart frames on an all-intra codec become a handful of
    short seek+decodes, while several nearby frames on a long-GOP codec become a
    single forward pass instead of one re-decoded GOP per frame.

    Frames are returned in the order requested (duplicates included). Returns
    ``None`` (caller falls back to a scan from the start) if a frame lacks a
    timestamp; raises :class:`IndexError` if an index is past the end of stream.
    """
    wanted = sorted(set(frame_indices))
    g = _seek_granularity(container, stream, wanted[-1], rate, time_base)
    if g is None:
        return None

    collected: dict[int, np.ndarray] = {}
    out_fmt = None
    decoder = None
    cur = None  # index of the most recently decoded frame
    for target in wanted:
        if cur is None or target < cur or target - cur > g:
            container.seek(int((target / rate) / time_base), stream=stream)
            decoder = container.decode(stream)
            cur = None
        reached = False
        for frame in decoder:
            if frame.pts is None:
                return None
            if out_fmt is None:
                out_fmt = _select_pixel_format(frame.format)
            cur = _index_at(time_base, rate, frame.pts)
            if cur >= target:
                collected[target] = frame.to_ndarray(format=out_fmt)
                reached = True
                break
        if not reached:
            raise IndexError(f"frame index {target} is out of range")

    return [collected[i] for i in frame_indices]


def read_frames_from_video(
    video_path: Path | str, frame_indices: list[int] | None = None
) -> tuple[list[np.ndarray], float | None]:
    """Read specific frames from a video file.

    Frames are decoded with PyAV at the source's **native bit depth and channel
    layout**: an 8-bit RGB video yields uint8 ``(H, W, 3)`` arrays, a 16-bit
    source (e.g. lossless FFV1) yields uint16 arrays, a source with an alpha
    channel yields 4-channel ``(H, W, 4)`` arrays, and a grayscale source yields
    2-D ``(H, W)`` arrays. This preserves the real pixel values; decoding such
    sources to 8-bit RGB (as imageio's reader does) would discard the low byte of
    every 16-bit sample and collapse small values to near-zero.

    Args:
        video_path: Path to the video file.
        frame_indices: Frame indices to read, in the order returned (duplicates
            allowed). If ``None``, reads all frames in order.

    Returns:
        A 2-tuple ``(frames, fps)``. *frames* is a list of numpy arrays whose
        dtype and channel count match the source (see above). *fps* is the
        average frame rate reported by the container, or ``None`` if unavailable.

    Raises:
        IndexError: If a requested frame index is out of range.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        fps = float(rate) if rate else None

        if frame_indices is not None and len(frame_indices) == 0:
            return [], fps

        # Fast path: specific frames + a usable frame rate. Seek to each frame
        # (or decode forward between nearby ones) instead of decoding the whole
        # stream; the strategy adapts to the codec's seek granularity.
        if frame_indices is not None and rate:
            result = _read_indexed_frames(
                container, stream, frame_indices, rate, stream.time_base
            )
            if result is not None:
                return result, fps

        # General path: a single forward decode, collecting wanted frames (or all
        # frames when frame_indices is None). Also the fallback when frames lack
        # timestamps; seek back to the start so indices map from frame 0.
        container.seek(0, stream=stream)
        target = None if frame_indices is None else set(frame_indices)
        out_fmt = None
        collected: dict[int, np.ndarray] = {}
        for i, frame in enumerate(container.decode(stream)):
            if out_fmt is None:
                out_fmt = _select_pixel_format(frame.format)
            if target is None or i in target:
                collected[i] = frame.to_ndarray(format=out_fmt)
                if target is not None and len(collected) == len(target):
                    break

    if frame_indices is None:
        return [collected[i] for i in sorted(collected)], fps
    missing = [i for i in frame_indices if i not in collected]
    if missing:
        raise IndexError(f"frame indices {sorted(set(missing))} are out of range")
    return [collected[i] for i in frame_indices], fps


def write_frames_to_video(
    video_path: Path | str,
    frames: list[np.ndarray],
    fps: float,
    *,
    mode: str = "auto",
    quality: int = _accel.DEFAULT_QUALITY,
    preset: str | None = None,
    extra_ffmpeg_params: list[str] | None = None,
    log_interval: int | None = None,
    quiet: bool = False,
) -> None:
    """Write a sequence of frames to an H.264 MP4 file.

    The encoder is chosen by *mode*: ``"gpu"`` encodes on the GPU (H.264 NVENC),
    ``"cpu"`` on the CPU (libx264), and ``"auto"`` (default) uses the GPU when a
    CUDA device and NVENC-capable FFmpeg are available, else the CPU. Either way
    the encode always falls back to libx264 if NVENC fails (e.g. frames below
    NVENC's minimum size), so output is always produced.

    Quality is controlled by the single *quality* knob, applied as libx264's CRF
    on the CPU path and as NVENC's constant QP on the GPU path — both on the same
    0-51 H.264 quantiser scale where lower means higher quality and larger files.
    The default of 20 is visually lossless and conservative, suitable for
    scientific data.

    Args:
        video_path: Path for the output video file.
        frames: Frames as uint8 numpy arrays in ``(H, W, C)`` format. All
            frames must share the same spatial dimensions.
        fps: Frames per second of the output video.
        mode: Encoder selection — ``"auto"`` (default), ``"gpu"``, or ``"cpu"``.
            ``"gpu"`` forces the NVENC path (falling back to libx264 if NVENC is
            unavailable for the input); ``"cpu"`` forces libx264; ``"auto"``
            picks the GPU when available.
        quality: Encode quality on the 0-51 H.264 quantiser scale (lower = higher
            quality, larger files). Applied as libx264's CRF or NVENC's QP
            depending on the chosen encoder, so it behaves consistently across
            the CPU and GPU paths.
        preset: Encoder preset. ``None`` uses a sensible per-encoder default
            (``"slow"`` for libx264, ``"p7"`` for NVENC). If given, it is passed
            to whichever encoder runs — use encoder-appropriate values
            (libx264: ``ultrafast``…``placebo``; NVENC: ``p1``…``p7``).
        extra_ffmpeg_params: Optional raw FFmpeg parameters appended after the
            quality/preset flags, as an escape hatch for advanced options.
        log_interval: If set, log progress every *log_interval* frames at
            ``INFO`` level.
        quiet: Suppress the encoder-parameters log line and the progress bar.
            When ``True``, *log_interval* is also ignored.

    Raises:
        ValueError: If *frames* is empty, contains frames with mismatched
            dimensions, or *mode* is not one of ``"auto"``/``"gpu"``/``"cpu"``.
    """
    if mode not in ("auto", "gpu", "cpu"):
        raise ValueError(f"mode must be 'auto', 'gpu', or 'cpu', got {mode!r}.")

    # Check frame size consistency
    if len(frames) == 0:
        raise ValueError("No frames provided to write_frames_to_video")
    frame_size = frames[0].shape[:2]
    for frame in frames:
        if frame.shape[:2] != frame_size:
            raise ValueError(
                "All frames must have the same dimensions. The 0th frame has size "
                f"{frame_size}, but at least one frame has size {frame.shape[:2]}."
            )
    height, width = frame_size[0], frame_size[1]

    _encode_with_fallback(
        video_path,
        frames,
        len(frames),
        height,
        width,
        fps,
        mode=mode,
        quality=quality,
        preset=preset,
        extra_ffmpeg_params=extra_ffmpeg_params,
        log_interval=log_interval,
        quiet=quiet,
    )


def write_image_paths_to_video(
    video_path: Path | str,
    image_paths: list[Path | str],
    fps: float,
    *,
    mode: str = "auto",
    quality: int = _accel.DEFAULT_QUALITY,
    preset: str | None = None,
    extra_ffmpeg_params: list[str] | None = None,
    log_interval: int | None = None,
    quiet: bool = False,
) -> None:
    """Combine on-disk image files into an H.264 MP4 file.

    Like :func:`write_frames_to_video`, but the frames are given as paths to
    image files (PNG, JPEG, TIFF, …) instead of in-memory numpy arrays. Images
    are read lazily, one at a time, so an arbitrarily long sequence can be
    encoded without holding every frame in memory at once. The encoder selection,
    quality semantics, and NVENC→libx264 fallback are identical to
    :func:`write_frames_to_video`.

    Frames are encoded in the order given by *image_paths*; sort the paths
    beforehand if a particular ordering is required. The first image's spatial
    dimensions define the output size, and every subsequent image must match it.

    Args:
        video_path: Path for the output video file.
        image_paths: Paths to the image files to combine, in output order. Each
            file is read with imageio and must decode to an ``(H, W, C)`` array;
            all images must share the same spatial dimensions.
        fps: Frames per second of the output video.
        mode: Encoder selection — ``"auto"`` (default), ``"gpu"``, or ``"cpu"``.
            See :func:`write_frames_to_video` for details.
        quality: Encode quality on the 0-51 H.264 quantiser scale (lower = higher
            quality, larger files). See :func:`write_frames_to_video`.
        preset: Encoder preset. ``None`` uses a sensible per-encoder default. See
            :func:`write_frames_to_video`.
        extra_ffmpeg_params: Optional raw FFmpeg parameters appended after the
            quality/preset flags.
        log_interval: If set, log progress every *log_interval* frames at
            ``INFO`` level.
        quiet: Suppress the encoder-parameters log line and the progress bar.
            When ``True``, *log_interval* is also ignored.

    Raises:
        ValueError: If *image_paths* is empty, an image's dimensions differ from
            the first image's, or *mode* is not one of
            ``"auto"``/``"gpu"``/``"cpu"``.

    A missing or unreadable image path surfaces whatever error imageio raises
    while reading it (typically :class:`FileNotFoundError`).
    """
    if mode not in ("auto", "gpu", "cpu"):
        raise ValueError(f"mode must be 'auto', 'gpu', or 'cpu', got {mode!r}.")

    paths = [Path(p) for p in image_paths]
    if len(paths) == 0:
        raise ValueError("No image paths provided to write_image_paths_to_video")

    # Read the first image to determine the output frame size. The remaining
    # images are validated lazily as they are read during encoding.
    first_frame = imageio.imread(paths[0])
    frame_size = first_frame.shape[:2]
    height, width = frame_size[0], frame_size[1]

    frames = _ImagePathFrames(paths, frame_size)
    _encode_with_fallback(
        video_path,
        frames,
        len(paths),
        height,
        width,
        fps,
        mode=mode,
        quality=quality,
        preset=preset,
        extra_ffmpeg_params=extra_ffmpeg_params,
        log_interval=log_interval,
        quiet=quiet,
    )


class _ImagePathFrames:
    """A re-iterable view over image files that decodes one frame at a time.

    Reading lazily keeps memory bounded regardless of how many frames are being
    combined. The view is re-iterable (a fresh generator per ``__iter__``) so the
    encoder's NVENC→libx264 fallback can restart the encode by re-reading from
    disk. Each frame's spatial dimensions are checked against *frame_size* as it
    is read, raising :class:`ValueError` on a mismatch.
    """

    def __init__(self, image_paths: list[Path], frame_size: tuple[int, int]):
        self._image_paths = image_paths
        self._frame_size = frame_size

    def __len__(self) -> int:
        return len(self._image_paths)

    def __iter__(self):
        for img_path in self._image_paths:
            frame = imageio.imread(img_path)
            if frame.shape[:2] != self._frame_size:
                raise ValueError(
                    "All frames must have the same dimensions. The first image "
                    f"has size {self._frame_size}, but {img_path} has size "
                    f"{frame.shape[:2]}."
                )
            yield frame


def _encode_with_fallback(
    video_path: Path | str,
    frames,
    n_frames: int,
    height: int,
    width: int,
    fps: float,
    *,
    mode: str,
    quality: int,
    preset: str | None,
    extra_ffmpeg_params: list[str] | None,
    log_interval: int | None,
    quiet: bool = False,
) -> None:
    """Encode *frames* with the NVENC→libx264 attempt/fallback strategy.

    Shared by :func:`write_frames_to_video` and
    :func:`write_image_paths_to_video`. *frames* is any re-iterable yielding
    uint8 ``(H, W, C)`` arrays (a list, or a lazy view such as
    :class:`_ImagePathFrames`); it may be iterated more than once when a GPU
    encode fails and the libx264 fallback restarts.
    """
    extra = list(extra_ffmpeg_params or [])

    # Decide whether to attempt NVENC. "auto" detects a usable GPU; "gpu" forces
    # it when the input/host allow; "cpu" stays on libx264.
    want_gpu = _accel.cuda_available() if mode == "auto" else (mode == "gpu")
    nvenc_usable = want_gpu and _accel.can_use_nvenc(height, width)
    if mode == "gpu" and not nvenc_usable:
        logger.warning(
            "mode='gpu' requested, but NVENC is unavailable for this input "
            "(no CUDA device / NVENC-capable FFmpeg, or frame too small); "
            "encoding with libx264 on the CPU instead."
        )

    # Build the ordered list of (codec, params, ffmpeg_exe) encode attempts. NVENC
    # is tried first when usable, always with a libx264 fallback so output is
    # produced even if the GPU encode fails. The libx264 fallback uses its own
    # default preset (a user-supplied preset is assumed NVENC-specific in that
    # case), but applies the requested quality.
    attempts: list[tuple[str, list[str], str | None]] = []
    if nvenc_usable:
        nvenc_preset = preset or _accel.DEFAULT_NVENC_PRESET
        attempts.append(
            (
                _accel.NVENC_CODEC,
                _accel.nvenc_params(quality, nvenc_preset) + extra,
                _accel.nvenc_ffmpeg_exe(),
            )
        )
        libx264_preset = _accel.DEFAULT_LIBX264_PRESET
    else:
        libx264_preset = preset or _accel.DEFAULT_LIBX264_PRESET
    attempts.append(
        (
            _accel.LIBX264_CODEC,
            _accel.libx264_params(quality, libx264_preset) + extra,
            None,
        )
    )

    last_error: Exception | None = None
    for attempt_idx, (attempt_codec, attempt_params, ffmpeg_exe) in enumerate(attempts):
        is_last = attempt_idx == len(attempts) - 1
        if not quiet:
            logger.info(
                "Encoder: %s | params: %s",
                attempt_codec,
                " ".join(attempt_params),
            )
        try:
            with _imageio_ffmpeg_exe(ffmpeg_exe):
                _encode_frames(
                    video_path,
                    frames,
                    n_frames,
                    fps,
                    attempt_codec,
                    attempt_params,
                    None if quiet else log_interval,
                    quiet=quiet,
                )
            return
        except Exception as e:
            last_error = e
            if is_last:
                raise
            logger.warning(
                "Encoding with codec %r failed (%s); falling back to %r.",
                attempt_codec,
                e,
                attempts[attempt_idx + 1][0],
            )
    # Unreachable: the last attempt either returns or re-raises.
    if last_error is not None:  # pragma: no cover - defensive
        raise last_error


def _encode_frames(
    video_path: Path | str,
    frames,
    n_frames: int,
    fps: float,
    codec: str,
    ffmpeg_params: list[str],
    log_interval: int | None,
    quiet: bool = False,
) -> None:
    """Encode *frames* to *video_path* with imageio's FFmpeg backend.

    *frames* is any iterable of uint8 ``(H, W, C)`` arrays; *n_frames* is its
    length, used only for progress logging.
    """
    use_tqdm = sys.stdout.isatty() and not quiet
    with imageio.get_writer(
        str(video_path),
        "ffmpeg",
        fps=fps,
        codec=codec,
        quality=None,  # Use CRF/QP (in ffmpeg_params) instead of quality
        ffmpeg_params=ffmpeg_params,
    ) as video_writer:
        frame_iter = (
            tqdm(frames, total=n_frames, unit="frame", dynamic_ncols=True)
            if use_tqdm
            else frames
        )
        for i, frame in enumerate(frame_iter):
            video_writer.append_data(frame)

            if (
                log_interval is not None
                and not use_tqdm
                and (i + 1) % log_interval == 0
            ):
                logger.info(f"Written frame {i + 1}/{n_frames}")


def _probe_video(video_path: Path | str) -> tuple[int, tuple[int, int], float | None]:
    """Read ``(n_frames, (height, width), fps)`` from a video without decoding.

    Uses PyAV (libav* bindings) to read the container/stream headers directly,
    so frame size and FPS are available instantly. The frame count comes from
    the stream header when present; otherwise it is obtained by *demuxing* the
    stream (counting compressed packets) rather than decoding frames.

    Counting via decode -- which is what ``imageio``'s ``count_frames()`` does by
    running ``ffmpeg -vf null -f null -`` -- is catastrophically slow on large
    lossless videos because every frame is fully decoded just to be discarded.
    Demuxing packets touches the same data ``ffprobe -count_packets`` does and
    stays sub-second even for multi-GB files.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        codec_ctx = stream.codec_context
        frame_size = (codec_ctx.height, codec_ctx.width)
        rate = stream.average_rate
        fps = float(rate) if rate else None

        num_frames = stream.frames
        if num_frames <= 0:
            # Header has no frame count (common for MKV). Count packets by
            # demuxing; this does NOT decode frames. The flush packet emitted at
            # end-of-stream has size 0, so it is excluded.
            num_frames = sum(1 for packet in container.demux(stream) if packet.size > 0)

    return num_frames, frame_size, fps


def check_num_frames(video_path: Path | str) -> int:
    """Return the number of frames in a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Total frame count.

    Raises:
        RuntimeError: If the file cannot be opened.
    """
    try:
        num_frames, _, _ = _probe_video(video_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open video file: {video_path}") from e
    return num_frames


def _file_signature(path: Path | str) -> tuple[int, int]:
    """Return a cheap ``(size_bytes, mtime_ns)`` fingerprint of *path*.

    Used as the staleness guard: if a file's size and modification time are
    unchanged since the cache was written, its contents are treated as unchanged.
    Any normal write (re-encode, truncate, append) bumps the size and/or mtime,
    so this is reliable in practice. The fingerprint is read from a single
    ``stat()`` call, so validation never touches the file contents -- important
    for multi-GB videos where hashing every byte would dominate the runtime.
    """
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def _read_metadata_cache(cache_path: Path, video_path: Path) -> VideoMetadata | None:
    """Load and validate the sidecar metadata cache.

    Returns the cached :class:`VideoMetadata` when the cache is present and its
    stored ``(size, mtime_ns)`` signature still matches the video, and ``None``
    to signal that the cache cannot be trusted and the metadata must be re-read
    (because the cache predates the signature format or the video has since been
    modified). Validation is a single ``stat()`` -- the video contents are never
    read here.

    Raises if the cache file exists but is structurally corrupt (unparseable
    JSON or a signature-validated payload missing required fields).
    """
    try:
        with open(cache_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.critical(f"Corrupted metadata cache file {cache_path}: {e}")
        raise

    if metadata.get("size") is None or metadata.get("mtime_ns") is None:
        logger.info(
            f"Metadata cache {cache_path} has no (size, mtime) signature (likely "
            f"written by an older version); re-reading metadata from the video."
        )
        return None

    size, mtime_ns = _file_signature(video_path)
    if metadata["size"] != size or metadata["mtime_ns"] != mtime_ns:
        logger.info(
            "Metadata cache %s: size/mtime differs from video; cache is stale and "
            "metadata will be re-read.",
            cache_path,
        )
        return None

    logger.info(
        "Metadata cache %s matches video signature; using cached metadata directly.",
        cache_path,
    )
    try:
        return VideoMetadata(
            n_frames=metadata["n_frames"],
            frame_size=tuple(metadata["frame_size"]),
            fps=metadata["fps"],
        )
    except Exception as e:
        logger.critical(f"Corrupted metadata cache file {cache_path}: {e}")
        raise


def _write_metadata_cache(
    cache_path: Path,
    video_path: Path,
    n_frames: int,
    frame_size: tuple[int, int],
    fps: float | None,
) -> None:
    """Atomically write the sidecar metadata cache.

    Stores the cheap ``(size, mtime_ns)`` signature alongside the metadata so
    :func:`_read_metadata_cache` can validate with a single ``stat()`` and never
    re-read the video's contents.
    """
    size, mtime_ns = _file_signature(video_path)
    metadata = {
        "n_frames": n_frames,
        "frame_size": list(frame_size),
        "fps": fps,
        "size": size,
        "mtime_ns": mtime_ns,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", dir=cache_path.parent, suffix=".tmp", delete=False
    ) as tmp_f:
        tmp_path = tmp_f.name
        json.dump(metadata, tmp_f, indent=2)
    os.replace(tmp_path, cache_path)


def get_video_metadata(
    video_path: Path | str,
    cache_metadata: bool = True,
    use_cached_metadata: bool = True,
    metadata_suffix: str = ".metadata.json",
) -> VideoMetadata:
    """Return frame count, frame size, and FPS for a video file.

    Results are cached to a sidecar JSON file alongside the video to avoid
    re-reading on subsequent calls. The cache stores the video's ``(size,
    mtime)`` signature and is automatically invalidated (the metadata is re-read
    and the cache rewritten) whenever that signature changes, i.e. whenever the
    video has been modified, so using the cache is safe in practice.

    Args:
        video_path: Path to the video file.
        cache_metadata: Write metadata to a cache file after reading.
        use_cached_metadata: Return cached metadata when the sidecar file exists
            and its ``(size, mtime)`` signature still matches the video. Set to
            ``False`` to force a fresh read regardless of the cache.
        metadata_suffix: Suffix appended to the video filename to form the
            cache path. Default: ``".metadata.json"``.

    Returns:
        A :class:`VideoMetadata` named tuple with fields ``n_frames`` (int),
        ``frame_size`` (``(height, width)`` tuple), and ``fps`` (float or
        ``None`` if unavailable).
    """
    video_path = Path(video_path)
    cache_path = video_path.parent / (video_path.name + metadata_suffix)

    if use_cached_metadata:
        if cache_path.is_file():
            cached = _read_metadata_cache(cache_path, video_path)
            if cached is not None:
                return cached
        else:
            logger.info(
                "No metadata cache found at %s; reading metadata from video.",
                cache_path,
            )

    n_frames, frame_size, fps = _probe_video(video_path)

    if cache_metadata:
        _write_metadata_cache(cache_path, video_path, n_frames, frame_size, fps)
        logger.info("Wrote metadata cache to %s.", cache_path)

    return VideoMetadata(n_frames=n_frames, frame_size=tuple(frame_size), fps=fps)
