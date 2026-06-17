"""Unit tests for VideoCollectionDataset and VideoCollectionDataLoader."""

import pytest
import torch
import numpy as np
import imageio.v2 as imageio
import tempfile
from pathlib import Path
from multiprocessing import cpu_count

from pvio.torch_tools import (
    VideoCollectionDataLoader,
    VideoCollectionDataset,
    SimpleVideoCollectionLoader,
    EncodedVideo,
    ImageDirVideo,
)
from pvio.io import write_frames_to_video
from .test_utils import make_frames_with_stride, DummyDataset


# Test utilities
def create_test_images(
    directory: Path, count: int, prefix: str = "frame", use_regex_pattern: bool = True
):
    """Helper to create test image files."""
    directory.mkdir(exist_ok=True)
    img = np.zeros((32, 32, 3), dtype=np.uint8)

    for i in range(count):
        if use_regex_pattern:
            filename = f"{prefix}_{i:03d}.png"
        else:
            # Use letters for non-regex testing
            filename = f"{chr(ord('a') + i % 26)}.png"
        imageio.imwrite(directory / filename, img)


def create_test_video(path: Path, n_frames: int = 15) -> Path:
    """Helper to create a test video file."""
    frames = make_frames_with_stride(n_frames)
    write_frames_to_video(path, frames, fps=5.0)
    return path


# Core functionality tests
def test_frame_extraction_parsing():
    """Test frame number extraction from filenames."""
    parse = ImageDirVideo._parse_frame_id_from_filename

    # Successful extractions
    assert parse("img_001.png", r"(\d+)") == 1
    assert parse("frame_42.jpg", r"(\d+)") == 42

    # Error cases
    with pytest.raises(ValueError, match="0 matches found"):
        parse("no_numbers.png", r"(\d+)")
    with pytest.raises(ValueError, match="2 matches found"):
        parse("img_01_02.png", r"(\d+)")


def test_dataloader_basic_functionality():
    """Test basic DataLoader functionality with collation."""
    ds = DummyDataset()
    loader = VideoCollectionDataLoader(ds, batch_size=3, num_workers=0)

    batch = next(iter(loader))
    assert "frames" in batch and "video_indices" in batch and "frame_indices" in batch
    assert batch["frames"].shape == (3, 3, 4, 5)

    # Check values are correctly collated
    assert torch.equal(batch["frames"][0], torch.zeros(3, 4, 5))
    assert torch.equal(batch["frames"][2], torch.ones(3, 4, 5) * 2)


def test_dataloader_validation():
    """Test DataLoader initialization validation."""
    with pytest.raises(ValueError, match="only works with VideoCollectionDataset"):
        VideoCollectionDataLoader(object(), batch_size=1)

    ds = DummyDataset()

    with pytest.raises(ValueError, match="does not support custom batch samplers"):
        VideoCollectionDataLoader(ds, batch_size=1, batch_sampler=[1])

    with pytest.raises(ValueError, match="must use the built-in collate function"):
        VideoCollectionDataLoader(ds, batch_size=1, collate_fn=lambda x: x)


def test_image_directory_handling(tmp_path: Path):
    """Test ImageDirVideo with different sorting strategies."""
    # Test filename-based sorting (no regex)
    d1 = tmp_path / "frames1"
    create_test_images(d1, 3, use_regex_pattern=False)

    video = ImageDirVideo(d1, frame_id_regex=None)
    video.setup()  # Need to call setup to populate phy_frame_id_to_path
    paths = [
        video.phy_frame_id_to_path[i] for i in sorted(video.phy_frame_id_to_path.keys())
    ]
    assert [p.name for p in paths] == ["a.png", "b.png", "c.png"]

    # Test regex-based sorting
    d2 = tmp_path / "frames2"
    create_test_images(d2, 3, "frame")

    video2 = ImageDirVideo(d2, frame_id_regex=r"(\d+)")
    video2.setup()  # Need to call setup to populate phy_frame_id_to_path
    paths2 = [
        video2.phy_frame_id_to_path[i]
        for i in sorted(video2.phy_frame_id_to_path.keys())
    ]
    assert [p.name for p in paths2] == [
        "frame_000.png",
        "frame_001.png",
        "frame_002.png",
    ]


def test_video_object_creation():
    """Test Video object parameter handling."""
    # EncodedVideo parameters
    dummy_path = Path("/fake/path.mp4")
    video = EncodedVideo(dummy_path, buffer_size=128)
    assert video.buffer_size == 128
    assert video.path == dummy_path

    # ImageDirVideo with real directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        create_test_images(tmp_path, 1, "frame")

        video2 = ImageDirVideo(tmp_path, frame_id_regex=r"(\d+)")
        assert video2.path == tmp_path


@pytest.mark.parametrize(
    "n_workers,expected_workers",
    [
        (0, 1),  # Zero workers becomes 1
    ],
)
def test_worker_assignment(tmp_path: Path, n_workers, expected_workers):
    """Test worker assignment with different configurations."""
    # Create test videos
    videos = []
    frame_counts = [25, 15, 30]

    for i, frames in enumerate(frame_counts):
        path = tmp_path / f"video_{i}.mp4"
        create_test_video(path, frames)
        videos.append(EncodedVideo(path))

    ds = VideoCollectionDataset(videos)
    ds.assign_workers(n_loading_workers=expected_workers)

    # Check basic properties
    assert ds.n_frames_total == sum(frame_counts)
    assert len(ds.worker_assignments) == expected_workers

    # Verify all frames are assigned
    total_assigned = sum(
        end - start for worker in ds.worker_assignments for _, start, end in worker
    )
    assert total_assigned == ds.n_frames_total


def test_worker_assignment_reduction(tmp_path: Path):
    """Test that workers are reduced when there aren't enough frames."""
    # Create small videos that will trigger worker reduction
    videos = []
    frame_counts = [25, 15, 30]  # Total 70 frames

    for i, frames in enumerate(frame_counts):
        path = tmp_path / f"video_{i}.mp4"
        create_test_video(path, frames)
        videos.append(EncodedVideo(path))

    ds = VideoCollectionDataset(videos)
    ds.assign_workers(n_loading_workers=2)  # Request 2 workers

    # With default min_frames_per_worker=300 and only 70 frames total,
    # should reduce to 1 worker
    # Each worker should still have a list of assignments
    assert len(ds.worker_assignments) == 2
    # ... but the assengment for the second worker should be empty
    assert len(ds.worker_assignments[1]) == 0

    # All frames should still be assigned
    total_assigned = sum(
        end - start for worker in ds.worker_assignments for _, start, end in worker
    )
    assert total_assigned == sum(frame_counts)


def test_simple_video_collection_loader(tmp_path: Path):
    """Test SimpleVideoCollectionLoader end-to-end functionality."""
    # Test with mixed video types
    video_file = tmp_path / "video.mp4"
    image_dir = tmp_path / "frames"

    create_test_video(video_file, 20)
    create_test_images(image_dir, 15, "frame")

    # Test basic functionality
    loader = SimpleVideoCollectionLoader(
        [video_file, image_dir],
        batch_size=8,
        num_workers=0,
        frame_id_regex=r"(\d+)",
        min_frames_per_worker=5,
    )

    # Should inherit from VideoCollectionDataLoader
    assert isinstance(loader, VideoCollectionDataLoader)
    assert isinstance(loader.dataset, VideoCollectionDataset)

    # Test iteration
    total_frames = 0
    video_indices_seen = set()

    for batch in loader:
        assert all(key in batch for key in ["frames", "video_indices", "frame_indices"])
        total_frames += len(batch["frame_indices"])
        video_indices_seen.update(batch["video_indices"])

    assert total_frames == 35  # 20 + 15
    assert video_indices_seen == {0, 1}  # Both videos represented


def test_worker_resolution():
    """Test worker count resolution logic."""
    from pvio.torch_tools import _resolve_n_workers_spec

    # Test automatic worker resolution
    loader = SimpleVideoCollectionLoader([], num_workers=-1, min_frames_per_worker=1)
    assert 0 < loader.num_workers <= cpu_count()

    # Test zero workers stays 0 (main-process DataLoader, no pickling)
    loader2 = SimpleVideoCollectionLoader([], num_workers=0, min_frames_per_worker=1)
    assert loader2.num_workers == 0

    # Bug 3: values above cpu_count() must be accepted (valid DataLoader use case)
    above_cpu = cpu_count() + 4
    assert _resolve_n_workers_spec(above_cpu) == above_cpu


def test_balanced_distribution_large_video(tmp_path: Path):
    """Test that large videos get balanced worker distribution."""
    # Create a large video
    large_video = tmp_path / "large.mp4"
    create_test_video(large_video, 600)

    video = EncodedVideo(large_video)
    ds = VideoCollectionDataset([video])

    # Force smaller min_frames_per_worker for testing
    ds.assign_workers(n_loading_workers=2, min_frames_per_worker=50)

    # Should have 2 workers, each with ~300 frames
    frames_per_worker = [
        sum(end - start for _, start, end in worker_assignments)
        for worker_assignments in ds.worker_assignments
    ]

    assert len(frames_per_worker) == 2
    assert frames_per_worker[0] == 300
    assert frames_per_worker[1] == 300


def test_small_video_worker_reduction(tmp_path: Path):
    """Test that small videos reduce worker count appropriately."""
    small_video = tmp_path / "small.mp4"
    create_test_video(small_video, 5)

    video = EncodedVideo(small_video)
    ds = VideoCollectionDataset([video])
    ds.assign_workers(n_loading_workers=4)  # Request more workers than frames

    # Should reduce to 1 worker due to minimum frames constraint
    # (assignment list should still exist for all requested workers, but some are empty)
    assert len(ds.worker_assignments) == 4
    # only 5 frames - should be assigned to the same worker
    assert len(ds.worker_assignments[0]) == 1
    for i in range(1, 4):
        assert len(ds.worker_assignments[i]) == 0

    total_frames = sum(end - start for _, start, end in ds.worker_assignments[0])
    assert total_frames == 5


def test_dataset_iter_without_assign_workers_raises(tmp_path: Path):
    """Bug 8: iterating VideoCollectionDataset without calling assign_workers (or using
    VideoCollectionDataLoader) must raise a clear, user-facing error."""
    from pvio.video import ImageDirVideo

    d = tmp_path / "frames"
    d.mkdir()
    import imageio.v2 as _iio

    _iio.imwrite(d / "frame_000.png", np.zeros((8, 8, 3), dtype=np.uint8))

    video = ImageDirVideo(d)
    ds = VideoCollectionDataset([video])

    with pytest.raises(RuntimeError, match="VideoCollectionDataLoader"):
        next(iter(ds))


def test_encoded_video_buffer_stores_pretransform(tmp_path: Path):
    """Bug 2: EncodedVideo buffer must cache raw frames, not post-transform frames.

    If the buffer stores the transformed frame, a second read of the same frame
    with a different transform returns the wrong result.
    """
    video_path = create_test_video(tmp_path / "vid.mp4", n_frames=5)
    video = EncodedVideo(video_path, buffer_size=10)
    video.setup()

    double = lambda x: x * 2.0  # noqa: E731
    half = lambda x: x * 0.5  # noqa: E731

    # First read populates the buffer
    frame_doubled = video.read_frame(1, transform=double)
    # Second read on the same frame is a cache hit
    frame_halved = video.read_frame(1, transform=half)

    # With bug: frame_halved == frame_doubled (transform baked into cache)
    # With fix: they differ because transforms are applied to the raw cached frame
    assert not torch.allclose(frame_doubled, frame_halved), (
        "Buffer is storing post-transform frames; second read ignores its transform"
    )


def test_image_dir_video_frame_range_offset(tmp_path: Path):
    """Bug 1: ImageDirVideo.read_frame(n) must return frame at physical position
    frame_range[0]+n, not position n."""
    d = tmp_path / "frames"
    d.mkdir()
    # 10 frames, each filled with a unique color value (0, 10, 20, ..., 90)
    for i in range(10):
        color = i * 10
        img = np.full((8, 8, 3), fill_value=color, dtype=np.uint8)
        imageio.imwrite(d / f"frame_{i:03d}.png", img)

    video = ImageDirVideo(d, frame_range=(3, 7))
    video.setup()

    assert len(video) == 4

    # Virtual index 0 → physical frame 3 → color=30
    frame0 = video.read_frame(0)
    assert abs(frame0.mean().item() - 30 / 255.0) < 0.01

    # Virtual index 3 → physical frame 6 → color=60
    frame3 = video.read_frame(3)
    assert abs(frame3.mean().item() - 60 / 255.0) < 0.01


# ---------------------------------------------------------------------------
# Comprehensive tests
# ---------------------------------------------------------------------------


class TestImageDirVideoComprehensive:
    """Comprehensive tests for ImageDirVideo."""

    def test_full_range_no_regex(self, tmp_path: Path):
        """No frame_range reads all frames in filename-sorted order."""
        d = tmp_path / "f"
        create_test_images(d, 5, use_regex_pattern=False)
        video = ImageDirVideo(d, frame_id_regex=None)
        video.setup()
        assert len(video) == 5

    def test_full_range_with_regex(self, tmp_path: Path):
        """No frame_range reads all frames when regex is used."""
        d = tmp_path / "f"
        create_test_images(d, 5)
        video = ImageDirVideo(d)
        video.setup()
        assert len(video) == 5

    def test_frame_range_length(self, tmp_path: Path):
        """frame_range=(a, b) exposes exactly b-a frames."""
        d = tmp_path / "f"
        create_test_images(d, 10)
        video = ImageDirVideo(d, frame_range=(2, 8))
        video.setup()
        assert len(video) == 6

    def test_frame_range_offset_no_regex(self, tmp_path: Path):
        """frame_range offset works correctly without regex (alphabetical sort)."""
        d = tmp_path / "f"
        d.mkdir()
        for i in range(5):
            img = np.full((8, 8, 3), fill_value=i * 50, dtype=np.uint8)
            imageio.imwrite(d / f"{chr(ord('a') + i)}.png", img)

        video = ImageDirVideo(d, frame_id_regex=None, frame_range=(1, 4))
        video.setup()
        # virtual 0 → 'b.png' → value 50
        frame = video.read_frame(0)
        assert abs(frame.mean().item() - 50 / 255.0) < 0.01

    def test_read_frame_returns_chw_float(self, tmp_path: Path):
        """read_frame returns a float tensor in CHW format in [0, 1]."""
        d = tmp_path / "f"
        create_test_images(d, 3)
        video = ImageDirVideo(d)
        video.setup()
        frame = video.read_frame(0)
        assert frame.dtype == torch.float32
        assert frame.ndim == 3  # CHW
        assert frame.min() >= 0.0
        assert frame.max() <= 1.0

    def test_read_frame_with_transform(self, tmp_path: Path):
        """Transform is applied to the loaded frame."""
        d = tmp_path / "f"
        d.mkdir()
        img = np.full((8, 8, 3), fill_value=100, dtype=np.uint8)
        imageio.imwrite(d / "frame_000.png", img)

        video = ImageDirVideo(d)
        video.setup()
        raw = video.read_frame(0)
        doubled = video.read_frame(0, transform=lambda x: x * 2.0)
        assert torch.allclose(doubled, raw * 2.0)

    def test_index_out_of_bounds_raises(self, tmp_path: Path):
        """Accessing a virtual index beyond the range raises IndexError."""
        d = tmp_path / "f"
        create_test_images(d, 3)
        video = ImageDirVideo(d, frame_range=(0, 2))
        video.setup()
        with pytest.raises(IndexError):
            video.read_frame(2)

    def test_setup_required_before_len(self, tmp_path: Path):
        """__len__ raises RuntimeError before setup() is called."""
        d = tmp_path / "f"
        create_test_images(d, 2)
        video = ImageDirVideo(d)
        with pytest.raises(RuntimeError):
            len(video)

    def test_16bit_image_normalized_by_dtype_max(self, tmp_path: Path):
        """16-bit images are normalized by 65535, not 255 (scientific TIFFs)."""
        d = tmp_path / "f16"
        d.mkdir()
        val = 30000
        img = np.full((8, 8), fill_value=val, dtype=np.uint16)
        imageio.imwrite(d / "frame_000.tif", img)

        video = ImageDirVideo(d)
        video.setup()
        frame = video.read_frame(0)
        assert frame.max().item() <= 1.0  # would be ~117 if divided by 255
        assert abs(frame.mean().item() - val / 65535.0) < 1e-3

    def test_grayscale_image_gets_channel_dim(self, tmp_path: Path):
        """Grayscale images are returned as (1, H, W) tensors."""
        d = tmp_path / "gray"
        d.mkdir()
        img = np.zeros((8, 8), dtype=np.uint8)  # 2D grayscale
        imageio.imwrite(d / "frame_000.png", img)

        video = ImageDirVideo(d)
        video.setup()
        frame = video.read_frame(0)
        assert frame.shape[0] == 1  # channel dim added

    def test_invalid_directory_raises(self):
        """Providing a nonexistent directory raises FileNotFoundError on setup."""
        video = ImageDirVideo("/nonexistent/path/to/dir")
        with pytest.raises(FileNotFoundError):
            video.setup()

    def test_frame_range_out_of_bounds_raises(self, tmp_path: Path):
        """frame_range that exceeds the available frames raises ValueError."""
        d = tmp_path / "f"
        create_test_images(d, 3)
        video = ImageDirVideo(d, frame_range=(0, 10))
        with pytest.raises(ValueError):
            video.setup()


class TestResolveNWorkersSpec:
    """Tests for _resolve_n_workers_spec edge cases."""

    def setup_method(self):
        from pvio.torch_tools import _resolve_n_workers_spec

        self.resolve = _resolve_n_workers_spec

    def test_positive_below_cpu_count(self):
        assert self.resolve(1) == 1

    def test_positive_above_cpu_count(self):
        """Values above cpu_count() are valid (Bug 3)."""
        n = cpu_count() + 8
        assert self.resolve(n) == n

    def test_zero_becomes_one(self):
        assert self.resolve(0) == 1

    def test_negative_one_uses_all_cores(self):
        assert self.resolve(-1) == cpu_count()

    def test_negative_two_uses_all_minus_one(self):
        if cpu_count() >= 2:
            assert self.resolve(-2) == cpu_count() - 1

    def test_too_negative_raises(self):
        with pytest.raises(ValueError):
            self.resolve(-(cpu_count() + 1))


class TestVideoCollectionDatasetWorkerAssignment:
    """Tests for worker assignment correctness."""

    def test_all_frames_covered(self, tmp_path: Path):
        """Every frame is assigned to exactly one worker."""
        paths = [tmp_path / f"v{i}.mp4" for i in range(3)]
        counts = [10, 15, 20]
        for path, n in zip(paths, counts):
            create_test_video(path, n)

        videos = [EncodedVideo(p) for p in paths]
        ds = VideoCollectionDataset(videos)
        ds.assign_workers(n_loading_workers=3)

        assigned = sum(e - s for worker in ds.worker_assignments for _, s, e in worker)
        assert assigned == sum(counts)

    def test_empty_workers_get_empty_assignment(self, tmp_path: Path):
        """Workers that receive no frames get an empty assignment list."""
        path = tmp_path / "tiny.mp4"
        create_test_video(path, 5)
        ds = VideoCollectionDataset([EncodedVideo(path)])
        ds.assign_workers(n_loading_workers=4)

        assert len(ds.worker_assignments) == 4
        assert any(len(a) == 0 for a in ds.worker_assignments)

    def test_assignment_indices_are_contiguous_within_video(self, tmp_path: Path):
        """Each worker's assignment for a given video is a contiguous range."""
        path = tmp_path / "big.mp4"
        create_test_video(path, 100)
        ds = VideoCollectionDataset([EncodedVideo(path)])
        ds.assign_workers(n_loading_workers=3, min_frames_per_worker=1)

        for worker_chunks in ds.worker_assignments:
            for _, start, end in worker_chunks:
                assert start < end


class TestVideoCollectionDataLoaderComprehensive:
    """End-to-end tests for VideoCollectionDataLoader."""

    def test_iteration_yields_all_frames(self, tmp_path: Path):
        """DataLoader iterates over every frame exactly once."""
        path = tmp_path / "v.mp4"
        create_test_video(path, 20)
        video = EncodedVideo(path)
        ds = VideoCollectionDataset([video])
        loader = VideoCollectionDataLoader(ds, batch_size=4, num_workers=0)

        total = sum(b["frames"].shape[0] for b in loader)
        assert total == 20

    def test_batch_tensor_shape(self, tmp_path: Path):
        """Each batch contains a (B, C, H, W) float32 tensor."""
        path = tmp_path / "v.mp4"
        create_test_video(path, 10)
        ds = VideoCollectionDataset([EncodedVideo(path)])
        loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=0)

        batch = next(iter(loader))
        frames = batch["frames"]
        assert frames.ndim == 4
        assert frames.dtype == torch.float32
        assert frames.min() >= 0.0
        assert frames.max() <= 1.0

    def test_video_and_frame_indices_in_batch(self, tmp_path: Path):
        """Batches contain matching video_indices and frame_indices."""
        path = tmp_path / "v.mp4"
        create_test_video(path, 6)
        ds = VideoCollectionDataset([EncodedVideo(path)])
        loader = VideoCollectionDataLoader(ds, batch_size=6, num_workers=0)

        batch = next(iter(loader))
        assert batch["video_indices"] == [0] * 6
        assert sorted(batch["frame_indices"]) == list(range(6))

    def test_multiple_videos_all_yielded(self, tmp_path: Path):
        """Frames from all videos appear in the iteration."""
        paths = [tmp_path / f"v{i}.mp4" for i in range(3)]
        for i, p in enumerate(paths):
            create_test_video(p, 10)

        videos = [EncodedVideo(p) for p in paths]
        ds = VideoCollectionDataset(videos)
        loader = VideoCollectionDataLoader(ds, batch_size=5, num_workers=0)

        total = sum(b["frames"].shape[0] for b in loader)
        assert total == 30

    def test_transform_applied_to_all_frames(self, tmp_path: Path):
        """Transform supplied to dataset is applied to every yielded frame."""
        d = tmp_path / "imgs"
        create_test_images(d, 4)
        ds = VideoCollectionDataset([ImageDirVideo(d)], transform=lambda x: x * 0.0)
        loader = VideoCollectionDataLoader(ds, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        assert batch["frames"].max() == 0.0


class TestSimpleVideoCollectionLoaderComprehensive:
    """Tests for SimpleVideoCollectionLoader."""

    def test_accepts_path_strings(self, tmp_path: Path):
        """Path strings are accepted alongside Path objects and Video instances."""
        path = tmp_path / "v.mp4"
        create_test_video(path, 5)
        loader = SimpleVideoCollectionLoader(
            [str(path)], batch_size=5, num_workers=0, min_frames_per_worker=1
        )
        total = sum(b["frames"].shape[0] for b in loader)
        assert total == 5

    def test_accepts_image_dir_path(self, tmp_path: Path):
        """Directory paths are automatically wrapped in ImageDirVideo."""
        d = tmp_path / "imgs"
        create_test_images(d, 6)
        loader = SimpleVideoCollectionLoader(
            [d],
            batch_size=6,
            num_workers=0,
            min_frames_per_worker=1,
            frame_id_regex=r"(\d+)",
        )
        total = sum(b["frames"].shape[0] for b in loader)
        assert total == 6

    def test_nonexistent_path_raises(self):
        """A path that does not exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SimpleVideoCollectionLoader(["/nonexistent/path.mp4"])

    def test_invalid_type_raises(self):
        """Passing an invalid type as a video spec raises TypeError."""
        with pytest.raises(TypeError):
            SimpleVideoCollectionLoader([42])
