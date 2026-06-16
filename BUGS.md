# Bug Report

Bugs found during audit of v0.1.5. Listed roughly in descending severity.

---

## BUG-1 — `ImageDirVideo` ignores `frame_range` in path lookup

**File:** `src/pvio/video.py`, `ImageDirVideo._post_setup` (lines 361–388)

`_post_setup` populates `vir_frame_id_to_path` with entries `0 .. N-1` for every file in the directory, regardless of `frame_range_effective`. As a result, `__len__` returns the correct (range-bounded) count, but `read_frame(n)` looks up the wrong file — it returns the frame at global sort position `n`, not at position `frame_range_effective[0] + n`.

**Example:**
```python
video = ImageDirVideo("dir_with_10_frames/", frame_range=(3, 7))
video.setup()
len(video)          # → 4  (correct)
video.read_frame(0) # → file at sort position 0, NOT position 3 (wrong)
```

`EncodedVideo` handles this correctly via `phy_frame_ids_to_buffer = vir_frame_ids_to_buffer + self.frame_range_effective[0]`. `ImageDirVideo` has no equivalent offset.

---

## BUG-2 — `EncodedVideo` buffer stores post-transform frames

**File:** `src/pvio/video.py`, `EncodedVideo._read_frame` (lines 259–276)

When the frame buffer is filled, each frame has `transform` applied before being stored. On a subsequent cache hit the stored (transformed) frame is returned directly, without considering the `transform` argument of the new call. If `_read_frame` is called twice for the same frame index with different transforms, the second call returns a frame that had the first transform applied.

This is benign under `VideoCollectionDataset`, which always passes the same transform. However, the public `read_frame` API accepts an arbitrary per-call transform, making the behaviour silently wrong in the general case.

---

## BUG-3 — `_resolve_n_workers_spec` rejects `n_workers > cpu_count()`

**File:** `src/pvio/torch_tools.py`, `_resolve_n_workers_spec` (lines 380–403)

The function raises `ValueError` for any `n_workers` greater than `cpu_count()`. PyTorch's `DataLoader` imposes no such restriction, and requesting more workers than cores is a legitimate configuration (e.g. to overlap I/O with compute). Users who pass `num_workers=cpu_count()+1` or any larger value will get an error that contradicts standard DataLoader expectations.

---

## BUG-4 — Metadata cache path replaces the video extension rather than appending

**File:** `src/pvio/io.py`, `get_video_metadata` (line 132)

```python
cache_path = video_path.with_suffix(metadata_suffix)
```

`Path.with_suffix` replaces the existing extension. `"video.mp4".with_suffix(".metadata.json")` → `"video.metadata.json"`, not `"video.mp4.metadata.json"`. Two videos with the same stem but different extensions (e.g. `clip.mp4` and `clip.avi`) map to the same cache file, causing silent metadata corruption for whichever is written second.

---

## BUG-5 — `fcntl` is imported unconditionally (Unix-only module)

**File:** `src/pvio/video.py`, line 13

```python
import fcntl
```

`fcntl` does not exist on Windows. The import is at module level, so `import pvio.video` (and by extension `import pvio`) fails immediately on Windows. `pyproject.toml` declares no platform restriction.

---

## BUG-6 — `write_frames_to_video` always logs at `i=0`

**File:** `src/pvio/io.py`, lines 97–98

```python
if log_interval is not None and i % log_interval == 0:
    logger.info(f"Written frame {i + 1}/{len(frames)}")
```

`0 % anything == 0`, so the very first frame is always logged regardless of `log_interval`. The intended behaviour (log every N frames) would be `(i + 1) % log_interval == 0`.

---

## BUG-7 — `ImageDirVideo._load_metadata` samples frame size from an arbitrary file

**File:** `src/pvio/video.py`, `ImageDirVideo._load_metadata` (lines 391–401)

`self.path.iterdir()` returns files in an OS-defined order. `all_files[0]` is whichever file the OS returns first, which may not be the first frame by name or index. If frames have mixed sizes, the `frame_size` reported by metadata (and stored on the `Video` object) may not reflect the majority or intended size. The practical risk is low for well-formed datasets, but the code should sort before sampling.

---

## BUG-8 — `VideoCollectionDataset.__iter__` gives a misleading error when `assign_workers` was never called

**File:** `src/pvio/torch_tools.py`, lines 192–196

If a caller uses `VideoCollectionDataset` directly with a standard `DataLoader` (bypassing `VideoCollectionDataLoader`) without calling `assign_workers` first, `self.worker_assignments` is `[]`. The check `len(self.worker_assignments) != 1` then raises:

> "Using a single worker but worker assignments indicate multiple workers."

Both the condition and the message are wrong in this case — there are zero assignments, not multiple. A missing-setup guard with a clear message would be more helpful.

---

## Minor / Documentation

- **`_post_setup` signature mismatch:** `ImageDirVideo._post_setup(self)` does not accept `*args/**kwargs`, but the base class `setup()` calls `self._post_setup(*args, **kwargs)`. Any subclass that passes extra args through `setup()` will hit a `TypeError`.
- **README cross-reference wrong module:** "Notes & troubleshooting" says to subclass `pvio.torch_tools.Video`, but `Video` lives in `pvio.video`.
