# Command-line interface

Installing the package puts a small `pvio` command on your `PATH` — a lightweight
ffmpeg helper for the two most common tasks. Run `pvio --help`, `pvio encode --help`,
or `pvio info --help` for the full set of options.

## `pvio encode` — combine image files into a video

Combine a directory (or an explicit list) of image files into an H.264 MP4. GPU
(NVENC) encoding is used automatically when available, falling back to the CPU
(libx264).

```bash
# Encode every image in a directory (natural-sorted: frame2 before frame10)
pvio encode frames/ --output out.mp4 --fps 30

# Encode an explicit, already-ordered list of files from a text file
pvio encode --from-file frame_paths.txt --output out.mp4 --no-sort

# Force the CPU encoder at a near-lossless quality
pvio encode frames/ --output out.mp4 --mode cpu --quality 18 --preset slow

# Force the GPU encoder (NVENC presets are p1…p7)
pvio encode frames/ --output out.mp4 --mode gpu --preset p7
```

Key options:

| Flag | Alias | Meaning |
| --- | --- | --- |
| `--output` | `-o` | Output `.mp4` path (required). Parent directories are created. |
| `--fps` | `-fps` | Frames per second (default: 30). |
| `--mode` | `-m` | `auto` (default), `gpu`, or `cpu`. |
| `--quality` | `-qa` | 0–51 H.264 quantiser scale, lower = higher quality (default: 20). |
| `--preset` | `-p` | Encoder preset — see the note below. |
| `--sort` / `--no-sort` | `-s` / `--no-s` | Natural numeric-aware ordering (on by default). |
| `--from-file` | `-f` | Text file with one image path per line (alternative to a directory). |
| `--quiet` / `--no-quiet` | `-q` / `--no-q` | Suppress encoder parameters, progress bar, and compression-ratio summary. |

!!! warning "Presets are encoder-specific"
    `--preset` must match the encoder that `--mode` selects: **libx264**
    (`ultrafast`…`placebo`) for `cpu`, and **NVENC** (`p1`…`p7`) for `gpu`.
    For libx264, faster presets encode more quickly but compress less efficiently.
    For NVENC, **lower numbers are faster but lower quality** (`p1` = fastest,
    `p7` = slowest / best quality). With an explicit `--mode`, a mismatched preset
    is a hard error; with `--mode auto`, the encoder is resolved best-effort and a
    mismatch is only a warning. Omit `--preset` to use a sensible per-encoder
    default (libx264 `slow`, NVENC `p7`).

## `pvio info` — inspect a video

Print a video's frame count, frame size, and FPS:

```bash
pvio info example.mp4
```

```text
path:       example.mp4
frames:     240
frame_size: 1920x1080 (width x height)
fps:        30
```

Pass `--no-cache` to force a fresh read instead of using (or writing) the sidecar
metadata cache.
