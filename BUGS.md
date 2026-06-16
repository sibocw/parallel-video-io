# Bug Report

Bugs found during audit of the current `dev-v0.1.6` tree. Listed roughly in
descending severity.

The full test suite (72 tests) passes, so none of these are caught by the
existing tests.

**Status (updated):**
- BUG-A / BUG-B / BUG-C — **fixed together** by a lock redesign: the per-instance
  `mkstemp` lock + `__del__` unlink were replaced with a single per-user shared lock
  file (`_DECODER_INIT_LOCK_PATH`) that is never unlinked. This removes the fd leak
  (A — no per-instance temp at all), actually serializes construction across all of a
  user's worker processes (B), and eliminates the deletion race (C). The original
  segfault could not be reproduced on the pinned stack (~2.5k concurrent
  constructions, zero crashes); the lock is kept only as cheap defensive insurance.
  See the reply on PR #7 for the full investigation.
- BUG-D — **resolved as documented behavior** (frame_range is positional by design;
  docstrings now state this explicitly).
- BUG-F — **fixed** (removed non-functional `*args`/`**kwargs` plumbing).
- BUG-G — **fixed** (worker-assignment indices cast to Python `int`).
- BUG-E, BUG-H — still open (not yet addressed).

---

## BUG-A — File-descriptor leak: `mkstemp()` handle never closed

**File:** `src/pvio/video.py`, `EncodedVideo.__init__` (lines 201–203)

```python
_, temp_path = mkstemp()
self._lock_path = Path(temp_path)
```

`mkstemp()` **opens** the temp file and returns an OS-level file descriptor as
its first element. That fd is discarded (`_`) and never closed, so every
`EncodedVideo` instance leaks one open file descriptor for the lifetime of the
process. The descriptor is unrelated to the locking logic — the lock is taken
later via a separate `open(lock_path, "w")` in `_create_video_decoder`.

**Confirmed:** creating 50 `EncodedVideo` objects raises the process fd count by
exactly 50. A collection of a few thousand videos (the intended use case) will
exhaust the default `ulimit -n` (often 1024) and fail with `OSError: [Errno 24]
Too many open files`, typically during `VideoCollectionDataset.__init__` where
all `EncodedVideo` objects are constructed up front.

**Fix direction:** close the fd immediately (`os.close(fd)`), or use
`tempfile.mkstemp` only for the name, or use `NamedTemporaryFile(delete=False)`
in a context manager.

---

## BUG-B — Per-video lock files do not serialize `VideoDecoder` construction across processes

**File:** `src/pvio/video.py`, `EncodedVideo.__init__` + `_create_video_decoder`
(lines 201–203, 266–280)

Each `EncodedVideo` creates its **own** lock file via `mkstemp()`
(confirmed: two videos get two different lock paths). The file lock in
`_create_video_decoder` is therefore per-video, but the race it is documented to
prevent is *global*:

> "FFmpeg initialises global state on first use; that initialisation is not safe
> across processes …"

In the normal parallel scenario, different DataLoader worker processes are
assigned **different** videos and each constructs its first `VideoDecoder`
concurrently — but for different videos, hence different lock files, hence **no
mutual exclusion**. The lock only serializes the rare case where two processes
construct a decoder for the *same* video object. As written, the mechanism does
not actually protect against the cross-process FFmpeg-init segfault it was added
to fix.

**Fix direction:** use a single, shared, fixed-path lock (one per machine, or one
per video *file* path) so that all processes serialize on the same lock during
decoder construction.

**Reference note (libav-user 2014-08):** the linked thread is about *thread*
safety, not *process* safety — global init (`avcodec_register_all`, lookup-table
setup) must be synchronized across **threads within a process**; across processes
the globals are not shared. DataLoader workers are separate processes and are
single-threaded during construction, so the documented race largely cannot occur
between them, and modern FFmpeg (≥4.0, used by torchcodec) made registration a
no-op / thread-safe. The lock's docstring rationale (thread-vs-process) is
inverted and should be corrected. Recommended: reproduce the original segfault on
the pinned stack first; if it reproduces, use one fixed shared lock that is never
unlinked (also fixes BUG-C); if not, remove the lock machinery entirely. Awaiting
maintainer decision.

---

## BUG-C — `__del__` unlinks the shared lock file used by sibling worker processes

**File:** `src/pvio/video.py`, `EncodedVideo.__del__` (lines 205–207)

```python
def __del__(self):
    self._lock_path.unlink(missing_ok=True)
```

DataLoader workers are forked, so all worker copies of an `EncodedVideo` share
the same `_lock_path` string. When the first worker to finish/garbage-collect
runs `__del__`, it deletes the lock file that other workers may still be using.
A later `open(lock_path, "w")` in another worker silently recreates the path as a
**new inode**, so `flock` on it no longer provides mutual exclusion with any
worker still holding the old inode. This compounds BUG-B and can re-introduce the
decoder-construction race. (Severity is bounded by the fact that decoders are
usually created early, before any worker exits.)

---

## BUG-D — `ImageDirVideo` `frame_range` indexes by sort position, not parsed frame id

**File:** `src/pvio/video.py`, `ImageDirVideo._post_setup` (lines 344–376)

The regex branch parses a physical frame id from each filename, sorts by it, then
applies `frame_range` to the **enumeration index** `sorted_idx`, not to the parsed
frame id:

```python
phy_frame_id_and_path.sort(key=lambda x: x[0])
for sorted_idx, (phy_frame_id, img_path) in enumerate(phy_frame_id_and_path):
    self.phy_frame_id_to_path[phy_frame_id] = img_path
    if start <= sorted_idx < end:      # <-- uses position, not phy_frame_id
        ...
```

For directories whose frame ids are sparse/non-contiguous this gives surprising
results. **Confirmed** with files `frame_000, frame_005, frame_010, frame_015,
frame_020` and `frame_range=(1, 3)`: virtual frame 0 maps to physical id **5**,
not id **1**. This is inconsistent with `EncodedVideo`, whose `frame_range`
refers to actual frame indices, and it makes the parsed `phy_frame_id` (the whole
point of the regex) irrelevant to range selection. The existing test
`test_image_dir_video_frame_range_offset` passes only because it uses contiguous
ids where `sorted_idx == phy_frame_id`.

**Note:** the same `start/end` from `_resolve_effective_frame_range` is validated
against `n_frames_total` (the file count), reinforcing that the intended unit is
"count/position." If position-based indexing is the deliberate semantic, it
should at least be documented; otherwise range selection should key off
`phy_frame_id`.

**Resolution:** positional indexing is the deliberate semantic (consistent with
`EncodedVideo` and the count-based length model). The `ImageDirVideo` class and
`_post_setup` docstrings now state this explicitly. No behavior change.

---

## BUG-E — `ImageDirVideo._load_metadata` samples `frame_size` from an arbitrary file

**File:** `src/pvio/video.py`, `ImageDirVideo._load_metadata` (lines 378–393)

```python
all_files = [f for f in self.path.iterdir() if f.is_file()]
...
sample_frame = imageio.imread(all_files[0])
frame_size = sample_frame.shape[:2]
```

`iterdir()` returns entries in OS-defined order, so `all_files[0]` is not the
first frame by sort order and may not be representative. For datasets with mixed
sizes the reported `frame_size` is non-deterministic. (Pre-existing, acknowledged
in a code comment; carried over from the previous audit's BUG-7 and still
unaddressed.) A related minor issue: any non-image file in the directory is
counted toward `n_frames_total` and can be picked as the sample, raising on
`imageio.imread`.

---

## BUG-F — `read_frame` forwards `*args` ahead of the `transform` keyword

**File:** `src/pvio/video.py`, `Video.read_frame` (lines 118–131)

```python
return self._read_frame(index, transform=transform, *args, **kwargs)
```

Python unpacks the positional `*args` into the parameter slots *after* `index` —
i.e. into the `transform` positional slot — while `transform=transform` is also
passed as a keyword. If any positional extra is ever supplied to `read_frame`,
this raises `TypeError: got multiple values for argument 'transform'`. The
concrete backends (`EncodedVideo._read_frame`, `ImageDirVideo._read_frame`) also
declare only `(self, index, transform)` with no `*args`/`**kwargs`, so the
documented "extra args" pass-through cannot work at all. Currently latent because
no caller passes extras.

---

## BUG-G — Batch `video_indices` / `frame_indices` contain NumPy ints, not Python ints

**File:** `src/pvio/torch_tools.py`, `assign_workers` (lines 168–175) and
`__iter__` (lines 208–212)

`np.unique(my_specs[:, 0])` yields `np.int32` `video_id` values, and
`start/end_vir_frame_id` are NumPy ints sliced from `frame_specs_all`. These flow
into the yielded dicts and the collated batch. The README and
`VideoCollectionDataLoader` docstring promise `video_indices`/`frame_indices` are
"list of int", but consumers receive `numpy.int32` objects. This can surprise
code that does strict `isinstance(x, int)` checks or JSON-serializes the indices.
Low severity but a documented-contract mismatch.

---

## BUG-H — Robustness: int32 frame index array and cache writes to read-only dirs

**Files:** `src/pvio/torch_tools.py` line 119; `src/pvio/io.py` lines 155–160

- `frame_specs_all = -np.ones((self.n_frames_total, 2), dtype=np.int32)` overflows
  for collections exceeding ~2.1B total frames. Unlikely in practice, but the type
  is needlessly narrow for a frame-index store.
- `get_video_metadata` writes its cache via `NamedTemporaryFile(dir=cache_path.parent, ...)`.
  If the video's directory is read-only, this raises even when the caller only
  wanted metadata. The cache write is best-effort and arguably should degrade
  gracefully (warn and skip) rather than abort the read.

Both are minor robustness concerns rather than functional bugs.
