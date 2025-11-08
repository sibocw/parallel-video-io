"""Unit tests for VideoCollectionLoader."""

import pytest
import numpy as np
from pathlib import Path

from pvio.torch import VideoCollectionLoader, _extract_frame_number
from pvio.video_io import write_frames_to_video

from .test_utils import make_frames_with_stride


def test_extract_frame_number_success_and_errors():
    """Test frame number extraction from filenames."""
    # Simple successful extraction
    assert _extract_frame_number("img_001.png", r"(\d+)") == 1

    # Multiple matches -> error
    with pytest.raises(ValueError):
        _extract_frame_number("img_01_02.png", r"(\d+)")

    # Non-integer match -> error
    with pytest.raises(ValueError):
        _extract_frame_number("frame_x.png", r"(x)")


def test_parse_n_workers():
    """Test worker count parsing with various inputs."""
    v1 = Path("dummy.mp4")  # Dummy path for initialization

    # Create a minimal loader just to test _parse_n_workers
    # We'll catch the ValueError from invalid path and that's okay
    import multiprocessing as mp

    cpu_count = mp.cpu_count()

    # Test with a mock loader object
    class MockLoader:
        def __init__(self):
            self.logger = VideoCollectionLoader.__init__.__globals__[
                "logging"
            ].getLogger(__name__)

    mock = MockLoader()
    parse_func = VideoCollectionLoader._parse_n_workers

    # Positive values
    assert parse_func(mock, 4) == 4
    assert parse_func(mock, 1) == 1

    # Zero
    assert parse_func(mock, 0) == 1  # Should warn and use 1

    # Negative values
    assert parse_func(mock, -1) == cpu_count  # All cores
    assert parse_func(mock, -2) == cpu_count - 1  # All but 1

    # Too high
    assert parse_func(mock, cpu_count + 10) == cpu_count  # Should cap

    # Invalid (too negative)
    with pytest.raises(ValueError):
        parse_func(mock, -cpu_count - 1)


def test_loader_initialization_validates_paths(tmp_path: Path):
    """Test that loader validates input paths."""
    # Invalid video file
    bad_video = tmp_path / "nonexistent.mp4"
    with pytest.raises(ValueError, match="not a valid file"):
        VideoCollectionLoader([bad_video], batch_size=4, n_loading_workers=1)

    # Invalid directory
    bad_dir = tmp_path / "nonexistent_dir"
    with pytest.raises(ValueError, match="not a valid directory"):
        VideoCollectionLoader(
            [bad_dir], batch_size=4, as_image_dirs=True, n_loading_workers=1
        )


def test_image_dirs_sorting_and_regex(tmp_path: Path):
    """Test image directory sorting with and without regex."""
    # Create files with unordered names
    d = tmp_path / "frames"
    d.mkdir()

    # Create dummy image files
    import imageio.v2 as imageio

    for name in ["b.png", "a.png", "c.png"]:
        img = np.full((10, 10, 3), fill_value=100, dtype=np.uint8)
        imageio.imwrite(d / name, img)

    loader = VideoCollectionLoader(
        [d], batch_size=3, as_image_dirs=True, n_loading_workers=1
    )

    # Check internal sorting (alphabetical)
    files = loader.sorted_frames_by_video_idx[0]
    assert [f.name for f in files] == ["a.png", "b.png", "c.png"]

    # Regex sorting
    d2 = tmp_path / "frames2"
    d2.mkdir()
    for name in ["frame_2.png", "frame_10.png", "frame_1.png"]:
        img = np.full((10, 10, 3), fill_value=100, dtype=np.uint8)
        imageio.imwrite(d2 / name, img)

    loader2 = VideoCollectionLoader(
        [d2],
        batch_size=3,
        as_image_dirs=True,
        frame_sorting=r"(\d+)",
        n_loading_workers=1,
    )

    files2 = loader2.sorted_frames_by_video_idx[0]
    assert [f.name for f in files2] == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_chunk_creation_with_fixed_size(tmp_path: Path):
    """Test that frames are split into chunks correctly."""
    v1 = tmp_path / "v1.mp4"
    write_frames_to_video(v1, make_frames_with_stride(25), fps=5.0)

    loader = VideoCollectionLoader(
        [v1], batch_size=5, chunk_size=10, n_loading_workers=1
    )

    # Should have 25 frames total
    assert loader.n_frames_total == 25

    # Should have 1 video
    assert len(loader.n_frames_by_video) == 1
    assert loader.n_frames_by_video[0] == 25


def test_chunk_creation_with_even_split(tmp_path: Path):
    """Test even_split chunk size calculation."""
    v1 = tmp_path / "v1.mp4"
    write_frames_to_video(v1, make_frames_with_stride(100), fps=5.0)

    n_workers = 4
    loader = VideoCollectionLoader(
        [v1], batch_size=10, chunk_size="even_split", n_loading_workers=n_workers
    )

    # With even_split and 4 workers, each should get ~25 frames
    # chunk_size = ceil(100 / 4) = 25
    assert loader.n_frames_total == 100


def test_multiple_videos_metadata_indexed(tmp_path: Path):
    """Test that multiple videos are indexed correctly."""
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"
    v3 = tmp_path / "v3.mp4"

    write_frames_to_video(v1, make_frames_with_stride(20), fps=5.0)
    write_frames_to_video(v2, make_frames_with_stride(15), fps=5.0)
    write_frames_to_video(v3, make_frames_with_stride(10), fps=5.0)

    loader = VideoCollectionLoader([v1, v2, v3], batch_size=5, n_loading_workers=2)

    # Check frame counts
    assert len(loader.n_frames_by_video) == 3
    assert loader.n_frames_by_video[0] == 20
    assert loader.n_frames_by_video[1] == 15
    assert loader.n_frames_by_video[2] == 10
    assert loader.n_frames_total == 45


def test_loader_length(tmp_path: Path):
    """Test __len__ returns correct number of batches."""
    v1 = tmp_path / "v1.mp4"
    write_frames_to_video(v1, make_frames_with_stride(27), fps=5.0)

    loader = VideoCollectionLoader([v1], batch_size=10, n_loading_workers=1)

    # 27 frames / 10 batch_size = 3 batches (27 = 10 + 10 + 7)
    assert len(loader) == 3

    loader2 = VideoCollectionLoader([v1], batch_size=5, n_loading_workers=1)
    # 27 frames / 5 batch_size = 6 batches (27 = 5*5 + 2)
    assert len(loader2) == 6


def test_paths_normalized_to_posix(tmp_path: Path):
    """Test that paths are normalized to POSIX strings."""
    v1 = tmp_path / "v1.mp4"
    write_frames_to_video(v1, make_frames_with_stride(10), fps=5.0)

    loader = VideoCollectionLoader([v1], batch_size=5, n_loading_workers=1)

    # Check that paths are stored as POSIX strings
    assert all(isinstance(p, str) for p in loader.video_paths_posix)
    assert loader.video_paths_posix[0] == v1.absolute().as_posix()
