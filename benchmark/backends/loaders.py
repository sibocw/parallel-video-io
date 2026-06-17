"""Multi-video parallel-loading backends — PVIO's headline use case.

Each backend streams every frame of a video *collection* through a batched
pipeline that delivers frames to the GPU, where a tiny consumer model runs (so
the GPU is actually exercised, as in real training/inference). We measure
aggregate end-to-end throughput, plus peak host/GPU memory, as a function of
worker count.

Backends:

* ``pvio``           — VideoCollectionDataLoader (frame-level balanced sharding).
* ``torchcodec_naive`` — hand-written IterableDataset that shards *whole videos*
  across workers (the common DIY approach), same TorchCodec decoder underneath.
* ``dali_gpu``       — NVIDIA DALI on-GPU video decode pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def make_consumer(device: str) -> nn.Module:
    """Tiny conv model standing in for a training/inference workload."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    ).to(device)
    model.eval()
    return model


class LoadingBackend:
    name = "base"

    def available(self) -> tuple[bool, str]:
        return True, ""

    def run(
        self, paths: list[str], num_workers: int, batch_size: int, device: str
    ) -> int:
        """Stream all frames once; return the number of frames consumed."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# PVIO
# --------------------------------------------------------------------------- #
class PvioLoader(LoadingBackend):
    name = "pvio"

    def run(self, paths, num_workers, batch_size, device) -> int:
        from pvio.torch_tools import SimpleVideoCollectionLoader

        model = make_consumer(device)
        loader = SimpleVideoCollectionLoader(
            paths,
            batch_size=batch_size,
            num_workers=num_workers,
            progress_bar=False,
            pin_memory=(device == "cuda"),
        )
        count = 0
        with torch.no_grad():
            for batch in loader:
                frames = batch["frames"].to(device, non_blocking=True)
                model(frames)
                count += frames.shape[0]
        if device == "cuda":
            torch.cuda.synchronize()
        return count


# --------------------------------------------------------------------------- #
# Naive TorchCodec IterableDataset (shards whole videos across workers)
# --------------------------------------------------------------------------- #
class _NaiveVideoDataset(IterableDataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __iter__(self):
        from torchcodec.decoders import VideoDecoder

        info = get_worker_info()
        if info is None:
            my_paths = self.paths
        else:
            my_paths = self.paths[info.id :: info.num_workers]
        for path in my_paths:
            decoder = VideoDecoder(path, seek_mode="approximate")
            for frame in decoder:  # CHW uint8
                yield frame.float() / 255.0


class TorchCodecNaiveLoader(LoadingBackend):
    name = "torchcodec_naive"

    def run(self, paths, num_workers, batch_size, device) -> int:
        model = make_consumer(device)
        dataset = _NaiveVideoDataset(paths)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )
        count = 0
        with torch.no_grad():
            for frames in loader:
                frames = frames.to(device, non_blocking=True)
                model(frames)
                count += frames.shape[0]
        if device == "cuda":
            torch.cuda.synchronize()
        return count


# --------------------------------------------------------------------------- #
# NVIDIA DALI (on-GPU decode)
# --------------------------------------------------------------------------- #
class DaliLoader(LoadingBackend):
    name = "dali_gpu"

    def available(self) -> tuple[bool, str]:
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        try:
            import nvidia.dali  # noqa: F401
        except Exception as e:  # pragma: no cover
            return False, f"import failed: {e}"
        return True, ""

    def run(self, paths, num_workers, batch_size, device) -> int:
        from nvidia.dali import fn, pipeline_def
        from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

        num_threads = max(1, num_workers)

        @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
        def pipe():
            return fn.readers.video(
                device="gpu",
                filenames=paths,
                sequence_length=1,
                random_shuffle=False,
                name="reader",
            )

        p = pipe()
        p.build()
        it = DALIGenericIterator(
            [p],
            ["frames"],
            reader_name="reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )
        model = make_consumer(device)
        count = 0
        with torch.no_grad():
            for batch in it:
                # (B, seq=1, H, W, C) uint8 on GPU -> (B, C, H, W) float
                frames = batch[0]["frames"][:, 0].permute(0, 3, 1, 2).float() / 255.0
                model(frames)
                count += frames.shape[0]
        torch.cuda.synchronize()
        return count


LOADING_BACKENDS: list[LoadingBackend] = [
    PvioLoader(),
    TorchCodecNaiveLoader(),
    DaliLoader(),
]
