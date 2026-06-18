"""Command-line interface for pvio.

A small, ffmpeg-replacement-style CLI for the two tasks people reach for most
often:

* ``pvio encode`` — combine a directory (or explicit list) of image files into
  an H.264 MP4, with GPU (NVENC) acceleration when available.
* ``pvio info`` — print a video's frame count, frame size, and FPS.

Run ``pvio --help``, ``pvio encode --help``, or ``pvio info --help`` for the
full set of options.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Annotated, Literal

import tyro

from . import _accel
from .io import get_video_metadata, write_image_paths_to_video


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


# File extensions treated as frame images when a directory is given to `encode`.
_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}
)

# Valid preset vocabularies, which differ per encoder: libx264 (CPU) uses named
# speed/quality presets, NVENC (GPU) uses the p1…p7 scale.
_LIBX264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)
_NVENC_PRESETS = ("p1", "p2", "p3", "p4", "p5", "p6", "p7")


def _natural_sort_key(path: Path) -> list:
    """Sort key that orders embedded numbers numerically (frame2 < frame10)."""
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", path.name)
    ]


def _collect_image_paths(inputs: list[Path], sort: bool) -> list[Path]:
    """Resolve *inputs* (image files and/or directories) to an ordered file list.

    A single directory is expanded to the image files it contains. Explicit file
    paths are kept in the given order unless *sort* is set, in which case the
    whole list is ordered with a natural (numeric-aware) sort.
    """
    if len(inputs) == 1 and inputs[0].is_dir():
        paths = [
            p
            for p in inputs[0].iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        ]
        if not paths:
            raise SystemExit(
                f"No image files found in directory {inputs[0]} "
                f"(looked for extensions: {', '.join(sorted(_IMAGE_EXTENSIONS))})."
            )
        return sorted(paths, key=_natural_sort_key)

    for p in inputs:
        if p.is_dir():
            raise SystemExit(
                f"{p} is a directory. Pass a single directory on its own, or a "
                f"list of image files (not a mix of directories and files)."
            )
    return sorted(inputs, key=_natural_sort_key) if sort else inputs


def _validate_preset(preset: str | None, mode: str) -> None:
    """Check ``--preset`` against the encoder that *mode* will use.

    libx264 (CPU) and NVENC (GPU) take different preset vocabularies, so a preset
    is only meaningful for the encoder that actually runs. For an explicit
    ``--mode`` the encoder is known, so a mismatch is a hard error (``SystemExit``).
    For ``--mode auto`` the encoder is resolved best-effort (NVENC needs a CUDA
    device and an NVENC-capable FFmpeg) and a mismatch is only a warning, since the
    real choice happens at encode time.
    """
    if preset is None:
        return

    if mode == "cpu":
        encoder, valid, certain = "libx264", _LIBX264_PRESETS, True
    elif mode == "gpu":
        encoder, valid, certain = "NVENC", _NVENC_PRESETS, True
    else:  # auto — best-effort guess at which encoder will run
        if _accel.cuda_available() and _accel.nvenc_ffmpeg_exe() is not None:
            encoder, valid, certain = "NVENC", _NVENC_PRESETS, False
        else:
            encoder, valid, certain = "libx264", _LIBX264_PRESETS, False

    if preset in valid:
        return

    message = (
        f"--preset {preset!r} is not a valid {encoder} preset. "
        f"Valid {encoder} presets: {', '.join(valid)}."
    )
    if certain:
        raise SystemExit(message)
    logging.warning(
        "%s mode=auto is expected to use %s on this machine, but the encoder is "
        "only finalised at encode time.",
        message,
        encoder,
    )


def encode(
    inputs: list[Path] = [],
    /,
    *,
    output: Annotated[Path, tyro.conf.arg(aliases=("-o",))],
    fps: Annotated[float, tyro.conf.arg(aliases=("-fps",))] = 30.0,
    mode: Annotated[
        Literal["auto", "gpu", "cpu"], tyro.conf.arg(aliases=("-m",))
    ] = "auto",
    quality: Annotated[int, tyro.conf.arg(aliases=("-qa",))] = _accel.DEFAULT_QUALITY,
    preset: Annotated[str | None, tyro.conf.arg(aliases=("-p",))] = None,
    sort: Annotated[bool, tyro.conf.arg(aliases=("-s",))] = True,
    from_file: Annotated[Path | None, tyro.conf.arg(aliases=("-f",))] = None,
    quiet: Annotated[bool, tyro.conf.arg(aliases=("-q",))] = False,
    log_interval: int | None = 100,
) -> None:
    """Combine image files into an H.264 MP4 video.

    Args:
        inputs: Image files to combine, or a single directory of images. A
            directory is expanded to its image files; an explicit list is
            encoded in the given order (then sorted, see ``sort``). Cannot be
            combined with ``--from-file``.
        output: Path for the output ``.mp4`` file.
        fps: Frames per second of the output video.
        mode: Encoder selection. ``auto`` uses the GPU (NVENC) when available
            and falls back to the CPU (libx264); ``gpu`` forces NVENC (still
            falling back to libx264 if unavailable); ``cpu`` forces libx264.
        quality: Quality on the 0-51 H.264 quantiser scale (lower = higher
            quality, larger files). Applied as libx264 CRF or NVENC QP.
        preset: Encoder preset. Omit for a sensible per-encoder default
            (libx264: ``slow``; NVENC: ``p7``). Must match the encoder that
            ``mode`` selects:

            - libx264 (CPU, ``--mode cpu``): ``ultrafast``, ``superfast``,
              ``veryfast``, ``faster``, ``fast``, ``medium``, ``slow``,
              ``slower``, ``veryslow``, ``placebo``. Faster presets encode
              more quickly but compress less efficiently (larger files at the
              same quality). Default ``slow`` is a good quality/speed balance.

            - NVENC (GPU, ``--mode gpu``): ``p1`` through ``p7``. Lower
              numbers are *faster* but produce lower quality / larger files;
              higher numbers are *slower* but achieve better quality. Default
              ``p7`` gives the best NVENC quality.

            A preset that doesn't match the active encoder is an error for an
            explicit ``--mode`` and a warning for ``--mode auto``.
        sort: Order frames with a natural (numeric-aware) sort. Always applied
            for a directory input; for an explicit list or ``--from-file``,
            disable with ``--no-sort`` to preserve the given order.
        from_file: Path to a plain-text file listing one image path per line.
            Use instead of positional ``inputs`` when encoding a large number
            of frames (avoids shell command-line length limits). Cannot be
            combined with positional ``inputs``.
        quiet: Suppress all informational output: encoder parameters, the
            progress bar, and the compression-ratio summary. Errors and
            warnings are still printed.
        log_interval: Log progress every N frames (when not in an interactive
            terminal). Set to a non-positive value to disable. Ignored when
            ``quiet`` is set.
    """
    if inputs and from_file is not None:
        raise SystemExit("Cannot use both positional inputs and --from-file.")
    if not inputs and from_file is None:
        raise SystemExit(
            "Provide image paths / a directory as positional arguments, "
            "or a text file of paths with --from-file."
        )

    if from_file is not None:
        if not from_file.is_file():
            raise SystemExit(f"{from_file} does not exist or is not a file.")
        lines = from_file.read_text().splitlines()
        raw_paths = [Path(ln.strip()) for ln in lines if ln.strip()]
        if not raw_paths:
            raise SystemExit(f"No paths found in {from_file}.")
        paths = sorted(raw_paths, key=_natural_sort_key) if sort else raw_paths
    else:
        paths = _collect_image_paths(inputs, sort)

    _validate_preset(preset, mode)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    input_bytes = sum(p.stat().st_size for p in paths)
    if not quiet:
        logging.info(
            "Encoding %d frame(s) into %s (mode=%s).", len(paths), output, mode
        )
    write_image_paths_to_video(
        output,
        paths,
        fps=fps,
        mode=mode,
        quality=quality,
        preset=preset,
        log_interval=log_interval if (log_interval and log_interval > 0) else None,
        quiet=quiet,
    )
    if not quiet:
        output_bytes = output.stat().st_size
        ratio = input_bytes / output_bytes if output_bytes else float("inf")
        logging.info(
            "Wrote %s  |  compression ratio %.1f× (%s → %s)",
            output,
            ratio,
            _fmt_bytes(input_bytes),
            _fmt_bytes(output_bytes),
        )


def info(
    video: Path,
    /,
    no_cache: bool = False,
) -> None:
    """Print a video's frame count, frame size, and FPS.

    Args:
        video: Path to the video file.
        no_cache: Force a fresh read instead of using (or writing) the sidecar
            metadata cache.
    """
    video = Path(video)
    if not video.is_file():
        raise SystemExit(f"{video} does not exist or is not a file.")

    metadata = get_video_metadata(
        video,
        cache_metadata=not no_cache,
        use_cached_metadata=not no_cache,
    )
    height, width = metadata.frame_size
    fps = "unknown" if metadata.fps is None else f"{metadata.fps:g}"
    print(f"path:       {video}")
    print(f"frames:     {metadata.n_frames}")
    print(f"frame_size: {width}x{height} (width x height)")
    print(f"fps:        {fps}")


def main() -> None:
    """Entry point for the ``pvio`` console script."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    tyro.extras.subcommand_cli_from_dict(
        {"encode": encode, "info": info},
        prog="pvio",
        description="Read, write, and inspect videos (an ffmpeg-lite helper).",
    )


if __name__ == "__main__":
    sys.exit(main())
