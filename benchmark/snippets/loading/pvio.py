# Build a parallel multi-video frame loader.
from pvio.torch_tools import SimpleVideoCollectionLoader


def make_loader(paths, batch_size, num_workers):
    return SimpleVideoCollectionLoader(
        paths, batch_size=batch_size, num_workers=num_workers
    )
