# Parallel Video IO

The Parallel Video IO (PVIO) package is motivated by the following problems that I kept having:

1. I could never remember the `ffmpeg` and `ffprob` commands for simple tasks, so I have to Google them every time.
2. Precise random seek in videos (for scientific use) is not so trivial.
3. I just want some simple dataloader that works for ML training _and_ inference.

After finding myself writing the same thing over and over again for different projects, I wrote this package with the following features:

1. Read frames from videos (random access or sequential) using imageio/FFmpeg.
2. Write sequences of NumPy frames to H.264 MP4 files with sensible defaults.
3. PyTorch-compatible `VideoCollectionDataset` and `VideoCollectionDataLoader` that stream frames from many videos in parallel across worker processes.
    - `SimpleVideoCollectionLoader` provides a convenience API that combines dataset and dataloader creation in one call.



## Notes & troubleshooting

**FFmpeg macroblock constraints.** Some FFmpeg builds require frame dimensions to be
divisible by 16. If you see a warning about `macro_block_size=16` and unexpected
resizing, use dimensions divisible by 16 in production pipelines.

**Metadata caching.** `get_video_metadata` writes a `.metadata.json` file next to each
video to speed up repeated indexing of large collections. Set `use_cached_metadata=False`
to force a fresh read.

**Custom backends.** Subclass `pvio.video.Video` and implement `_validate_init_params`,
`_load_metadata`, and `_read_frame` to add a custom video backend. See the
[Video Backends](api/video.md) API reference for the full subclassing contract.
