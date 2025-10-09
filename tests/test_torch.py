import pytest
import torch

from pvio.torch import VideoCollectionDataLoader, VideoCollectionDataset
from pvio.video_io import write_frames_to_video
import numpy as np
from pathlib import Path


def test_extract_frame_number_success_and_errors():
    # simple successful extraction
    assert VideoCollectionDataset._extract_frame_number("img_001.png", r"(\d+)") == 1

    # multiple matches -> error
    with pytest.raises(ValueError):
        VideoCollectionDataset._extract_frame_number("img_01_02.png", r"(\d+)")

    # non-integer match -> error
    with pytest.raises(ValueError):
        VideoCollectionDataset._extract_frame_number("frame_x.png", r"(x)")


class DummyDataset(VideoCollectionDataset):
    def __init__(self):
        # bypass parent initialization checks
        self.video_paths = []

    def assign_workers(
        self, n_frame_loading_workers: int, n_metadata_indexing_workers: int = -1
    ):
        # no-op for tests to avoid expensive metadata indexing
        return

    def __iter__(self):
        # yield three fake frame dicts
        for i in range(3):
            yield {
                "frame": torch.ones(3, 4, 5) * i,
                "video_path": f"video_{i}",
                "frame_idx": i,
            }


def test_collate_stacks_frames_correctly():
    ds = DummyDataset()
    # Create loader with zero workers to avoid assign_workers side-effects
    loader = VideoCollectionDataLoader(ds, batch_size=3, num_workers=0)

    batch = next(iter(loader))
    assert "frames" in batch and "video_paths" in batch and "frame_indices" in batch
    assert batch["frames"].shape == (3, 3, 4, 5)
    # check values
    assert torch.equal(batch["frames"][0], torch.zeros(3, 4, 5))
    assert torch.equal(batch["frames"][2], torch.ones(3, 4, 5) * 2)


def test_dataloader_init_type_and_kwargs_validation():
    # wrong dataset type
    with pytest.raises(ValueError):
        VideoCollectionDataLoader(object(), batch_size=1)

    # custom batch_sampler not allowed
    ds = DummyDataset()
    with pytest.raises(ValueError):
        VideoCollectionDataLoader(ds, batch_size=1, batch_sampler=[1])

    # custom collate_fn not allowed
    with pytest.raises(ValueError):
        VideoCollectionDataLoader(ds, batch_size=1, collate_fn=lambda x: x)


def test_vcd_as_image_dirs_sorting_and_regex(tmp_path: Path):
    # create files with unordered names
    d = tmp_path / "frames"
    d.mkdir()
    (d / "b.png").write_bytes(b"x")
    (d / "a.png").write_bytes(b"x")
    (d / "c.png").write_bytes(b"x")

    ds = VideoCollectionDataset([d], as_image_dirs=True)
    # inspect internal sorting
    files = ds.frame_sortings[d]
    assert [f.name for f in files] == ["a.png", "b.png", "c.png"]

    # regex sorting
    d2 = tmp_path / "frames2"
    d2.mkdir()
    (d2 / "frame_2.png").write_bytes(b"x")
    (d2 / "frame_10.png").write_bytes(b"x")
    (d2 / "frame_1.png").write_bytes(b"x")

    ds2 = VideoCollectionDataset([d2], as_image_dirs=True, frame_sorting=r"(\d+)")
    files2 = ds2.frame_sortings[d2]
    assert [f.name for f in files2] == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_assign_workers_splits_by_frame_count(tmp_path: Path):
    # create two small videos with different frame counts
    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"

    def make_frames(n):
        return [np.full((32, 32, 3), fill_value=i, dtype=np.uint8) for i in range(n)]

    write_frames_to_video(v1, make_frames(2), fps=5.0)
    write_frames_to_video(v2, make_frames(6), fps=5.0)

    ds = VideoCollectionDataset([v1, v2], as_image_dirs=False)
    ds.assign_workers(n_frame_loading_workers=2, n_metadata_indexing_workers=1)

    # both videos should be present in n_frames_lookup
    assert v1 in ds.n_frames_lookup and v2 in ds.n_frames_lookup
    # assignments should include all videos once
    flattened = [p for worker in ds.worker_assignments for p in worker]
    assert set(flattened) == set(ds.video_paths)
