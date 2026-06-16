# Developer Info

Developers should clone this repository and install the package via uv with `uv sync --extra dev`. The `dev` optional dependency installs packages for documentation site generation, testing, and formatting.

## Testing

The test suite uses pytest. Run it from the repository root:

```bash
pytest tests
```

The tests are organised into:

- `test_io.py` — tests for video I/O functions
- `test_torch_tools.py` — unit tests for `VideoCollectionDataset`
- `test_integration.py` — integration tests with parallel loading
- `test_readme_examples.py` — verifies the examples in the docs work end-to-end

A few tests write small MP4 files via imageio/FFmpeg; make sure `ffmpeg` is on your `$PATH`.


## Code style

- Use [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- In `.py` files, use ASCII characters only in docstrings and comments.
- Be concise in documentation.
- Use [ruff](https://docs.astral.sh/ruff/) for code formatting.
- Upon new release on GitHub, the package is pushed to PyPI and docs are pushed to the docs site via GitHub Pages.
