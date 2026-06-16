# Installation

!!! warning "Linux only"
    This package is currently tested on Linux only.

## Install with `pip`

```bash
pip install parallel-video-io
```

## Install by specifying `parallel-video-io` as dependency

Add `parallel-video-io` to you `pyproject.toml` file:

```toml
[project]
...  # other stuff
dependencies = [
    ...  # other stuff
    "parallel-video-io>=0.1.6",  # change version requirement accordingly
]
```

## For developers

Clone the repository and install it with [uv](https://docs.astral.sh/uv/):

```bash
git clone git@github.com:sibocw/parallel-video-io.git
cd parallel-video-io
uv sync  # developers should run `uv sync --extra dev`
```

Make sure `ffmpeg` is available on your `$PATH` (required by imageio-ffmpeg). On Debian/Ubuntu, this can be done with `sudo apt-get install ffmpeg`.
