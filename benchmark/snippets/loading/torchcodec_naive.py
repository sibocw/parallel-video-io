# Build a parallel multi-video frame loader by hand (shard videos over workers).
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchcodec.decoders import VideoDecoder


class VideoFrames(IterableDataset):
    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        info = get_worker_info()
        if info is None:
            my_paths = self.paths
        else:
            my_paths = self.paths[info.id :: info.num_workers]
        for path in my_paths:
            for frame in VideoDecoder(path, seek_mode="approximate"):
                yield frame.float() / 255.0


def make_loader(paths, batch_size, num_workers):
    return DataLoader(
        VideoFrames(paths), batch_size=batch_size, num_workers=num_workers
    )
