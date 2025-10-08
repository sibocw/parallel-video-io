import pytest
import torch

from pvio.torch import VideoCollectionDataLoader, VideoCollectionDataset


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
