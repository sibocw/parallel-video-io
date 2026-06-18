"""Tests for the pvio command-line interface."""

import numpy as np
import imageio.v2 as imageio
import pytest
from pathlib import Path

from pvio import cli
from pvio.io import get_video_metadata


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
