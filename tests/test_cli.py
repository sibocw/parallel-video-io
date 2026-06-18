"""Tests for the pvio command-line interface."""

import logging

import numpy as np
import imageio.v2 as imageio
import pytest
import tyro
from pathlib import Path

from pvio import cli
from pvio.io import get_video_metadata


def _invoke_cli(args: list[str]):
    """Run the pvio subcommand CLI with *args* (no sys.argv patching needed)."""
    return tyro.extras.subcommand_cli_from_dict(
        {"encode": cli.encode, "info": cli.info},
        args=args,
    )


def _make_images(dir_path: Path, names: list[str], h: int = 16, w: int = 16):
    dir_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, name in enumerate(names):
        arr = np.full((h, w, 3), (i * 20) % 256, dtype=np.uint8)
        p = dir_path / name
        imageio.imwrite(p, arr)
        paths.append(p)
    return paths


def test_natural_sort_orders_numbers_numerically(tmp_path: Path):
    """frame2 must sort before frame10 (numeric, not lexicographic)."""
    paths = _make_images(
        tmp_path, ["frame1.png", "frame2.png", "frame10.png", "frame11.png"]
    )
    # Shuffle the input order to ensure sorting actually happens.
    ordered = cli._collect_image_paths(list(reversed(paths)), sort=True)
    assert [p.name for p in ordered] == [
        "frame1.png",
        "frame2.png",
        "frame10.png",
        "frame11.png",
    ]


def test_collect_from_directory_filters_and_sorts(tmp_path: Path):
    """A single directory is expanded to its image files, non-images ignored."""
    img_dir = tmp_path / "frames"
    _make_images(img_dir, ["b10.png", "a2.png", "a1.png"])
    (img_dir / "notes.txt").write_text("ignore me")

    collected = cli._collect_image_paths([img_dir], sort=True)
    names = [p.name for p in collected]
    assert "notes.txt" not in names
    assert names == ["a1.png", "a2.png", "b10.png"]


def test_collect_empty_directory_raises(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        cli._collect_image_paths([empty], sort=True)


def test_collect_mixed_dir_and_files_raises(tmp_path: Path):
    img_dir = tmp_path / "frames"
    paths = _make_images(img_dir, ["a1.png"])
    with pytest.raises(SystemExit):
        cli._collect_image_paths([img_dir, paths[0]], sort=True)


def test_encode_produces_playable_video(tmp_path: Path):
    """The encode command writes a video with the expected frame count/size."""
    img_dir = tmp_path / "frames"
    _make_images(img_dir, [f"frame{i}.png" for i in range(5)], h=32, w=48)
    out = tmp_path / "nested" / "out.mp4"  # also exercises parent-dir creation

    cli.encode([img_dir], output=out, fps=10.0, mode="cpu")

    assert out.is_file()
    meta = get_video_metadata(out, cache_metadata=False, use_cached_metadata=False)
    assert meta.n_frames == 5
    assert meta.frame_size == (32, 48)


def test_info_prints_metadata(tmp_path: Path, capsys):
    img_dir = tmp_path / "frames"
    _make_images(img_dir, [f"f{i}.png" for i in range(3)], h=32, w=48)
    out = tmp_path / "clip.mp4"
    cli.encode([img_dir], output=out, fps=15.0, mode="cpu")

    cli.info(out)
    captured = capsys.readouterr().out
    assert "frames:     3" in captured
    assert "48x32" in captured  # width x height


def test_info_missing_file_raises(tmp_path: Path):
    with pytest.raises(SystemExit):
        cli.info(tmp_path / "does_not_exist.mp4")


# ---------------------------------------------------------------------------
# Preset / mode validation
# ---------------------------------------------------------------------------


def test_preset_none_always_ok():
    cli._validate_preset(None, "cpu")
    cli._validate_preset(None, "gpu")
    cli._validate_preset(None, "auto")


def test_explicit_cpu_with_libx264_preset_ok():
    cli._validate_preset("slow", "cpu")  # no raise


def test_explicit_gpu_with_nvenc_preset_ok():
    cli._validate_preset("p5", "gpu")  # no raise


def test_explicit_cpu_with_nvenc_preset_errors():
    with pytest.raises(SystemExit):
        cli._validate_preset("p7", "cpu")


def test_explicit_gpu_with_libx264_preset_errors():
    with pytest.raises(SystemExit):
        cli._validate_preset("slow", "gpu")


def test_auto_mismatch_warns_not_errors(monkeypatch, caplog):
    """In auto mode a mismatched preset warns rather than aborting."""
    # Force the best-effort resolution to CPU (libx264).
    monkeypatch.setattr(cli._accel, "cuda_available", lambda: False)
    with caplog.at_level("WARNING"):
        cli._validate_preset("p7", "auto")  # NVENC preset, but resolves to libx264
    assert any("not a valid libx264 preset" in r.message for r in caplog.records)


def test_auto_match_no_warning(monkeypatch, caplog):
    monkeypatch.setattr(cli._accel, "cuda_available", lambda: False)
    with caplog.at_level("WARNING"):
        cli._validate_preset("slow", "auto")
    assert not caplog.records


def test_encode_rejects_mismatched_preset_before_work(tmp_path: Path):
    """A bad explicit-mode preset aborts before any image is read/written."""
    img_dir = tmp_path / "frames"
    _make_images(img_dir, ["frame0.png"])
    out = tmp_path / "out.mp4"
    with pytest.raises(SystemExit):
        cli.encode([img_dir], output=out, mode="cpu", preset="p7")
    assert not out.exists()


# ---------------------------------------------------------------------------
# quiet flag
# ---------------------------------------------------------------------------


def test_encode_quiet_suppresses_info_logs(tmp_path: Path, caplog):
    """quiet=True emits no INFO-level messages."""
    img_dir = tmp_path / "frames"
    _make_images(img_dir, [f"frame{i}.png" for i in range(3)], h=16, w=16)
    out = tmp_path / "out.mp4"

    with caplog.at_level(logging.INFO):
        cli.encode([img_dir], output=out, fps=10.0, mode="cpu", quiet=True)

    info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_msgs == [], (
        f"Expected no INFO logs with quiet=True, got: {[r.message for r in info_msgs]}"
    )


def test_encode_quiet_still_produces_valid_video(tmp_path: Path):
    """quiet=True does not affect correctness of the output video."""
    img_dir = tmp_path / "frames"
    _make_images(img_dir, [f"frame{i}.png" for i in range(4)], h=32, w=48)
    out = tmp_path / "out.mp4"

    cli.encode([img_dir], output=out, fps=10.0, mode="cpu", quiet=True)

    assert out.is_file()
    meta = get_video_metadata(out, cache_metadata=False, use_cached_metadata=False)
    assert meta.n_frames == 4
    assert meta.frame_size == (32, 48)


def test_encode_quiet_warnings_still_emitted(monkeypatch, caplog):
    """quiet=True should not suppress WARNING-level messages."""
    monkeypatch.setattr(cli._accel, "cuda_available", lambda: False)
    with caplog.at_level(logging.WARNING):
        cli._validate_preset("p7", "auto")
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# Short CLI aliases
# ---------------------------------------------------------------------------


def test_cli_short_alias_output(tmp_path: Path):
    """-o accepted as alias for --output."""
    img_dir = tmp_path / "frames"
    _make_images(img_dir, ["frame0.png", "frame1.png"], h=16, w=16)
    out = tmp_path / "out.mp4"

    _invoke_cli(["encode", str(img_dir), "-o", str(out), "--mode", "cpu", "-q"])
    assert out.is_file()


def test_cli_short_alias_fps(tmp_path: Path):
    """-fps accepted as alias for --fps."""
    img_dir = tmp_path / "frames"
    _make_images(img_dir, [f"frame{i}.png" for i in range(3)], h=16, w=16)
    out = tmp_path / "out.mp4"

    _invoke_cli(
        [
            "encode",
            str(img_dir),
            "--output",
            str(out),
            "-fps",
            "24",
            "--mode",
            "cpu",
            "-q",
        ]
    )

    meta = get_video_metadata(out, cache_metadata=False, use_cached_metadata=False)
    assert meta.fps == pytest.approx(24.0, abs=1.0)


def test_cli_short_alias_mode(tmp_path: Path):
    """-m accepted as alias for --mode."""
    img_dir = tmp_path / "frames"
    _make_images(img_dir, ["frame0.png"], h=16, w=16)
    out = tmp_path / "out.mp4"

    _invoke_cli(["encode", str(img_dir), "-o", str(out), "-m", "cpu", "-q"])
    assert out.is_file()


def test_cli_short_alias_quality(tmp_path: Path):
    """-qa accepted as alias for --quality."""
    img_dir = tmp_path / "frames"
    _make_images(img_dir, ["frame0.png"], h=16, w=16)
    out = tmp_path / "out.mp4"

    _invoke_cli(
        ["encode", str(img_dir), "-o", str(out), "--mode", "cpu", "-qa", "28", "-q"]
    )
    assert out.is_file()


def test_cli_short_alias_from_file(tmp_path: Path):
    """-f accepted as alias for --from-file."""
    img_dir = tmp_path / "frames"
    paths = _make_images(img_dir, [f"frame{i}.png" for i in range(3)], h=16, w=16)
    list_file = tmp_path / "paths.txt"
    list_file.write_text("\n".join(str(p) for p in paths))
    out = tmp_path / "out.mp4"

    _invoke_cli(["encode", "-f", str(list_file), "-o", str(out), "--mode", "cpu", "-q"])

    meta = get_video_metadata(out, cache_metadata=False, use_cached_metadata=False)
    assert meta.n_frames == 3


def test_cli_short_alias_quiet(tmp_path: Path, caplog):
    """-q accepted as alias for --quiet and suppresses INFO logs."""
    img_dir = tmp_path / "frames"
    _make_images(img_dir, ["frame0.png", "frame1.png"], h=16, w=16)
    out = tmp_path / "out.mp4"

    with caplog.at_level(logging.INFO):
        _invoke_cli(["encode", str(img_dir), "-o", str(out), "--mode", "cpu", "-q"])

    info_msgs = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_msgs == []


def test_cli_short_alias_sort(tmp_path: Path):
    """-s accepted as alias for --sort."""
    img_dir = tmp_path / "frames"
    _make_images(img_dir, ["frame1.png", "frame2.png"], h=16, w=16)
    out = tmp_path / "out.mp4"

    _invoke_cli(["encode", str(img_dir), "-o", str(out), "--mode", "cpu", "-s", "-q"])
    assert out.is_file()
