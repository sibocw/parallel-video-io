# `pvio.video` — Video Backends

Abstract base class and concrete backends for video sources.

`Video` is the base class for all video sources. Subclass it and implement
`_validate_init_params`, `_load_metadata`, and `_read_frame` to add a custom
backend.

::: pvio.video.Video
    options:
      filters: []
      members:
        - __init__
        - setup
        - read_frame
        - close
        - _validate_init_params
        - _load_metadata
        - _read_frame
        - _post_setup

::: pvio.video.EncodedVideo

::: pvio.video.ImageDirVideo
