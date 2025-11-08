import torch
import logging
import re
import imageio.v2 as imageio
import numpy as np
import multiprocessing as mp
import queue
from typing import Callable, Literal
from torchcodec.decoders import VideoDecoder
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from .video_io import get_video_metadata


class VideoCollectionLoader:
    """Yields individual frames from Spotlight behavior recordings, either
    from videos or from image sequences."""

    def __init__(
        self,
        paths: list[Path | str],
        batch_size: int,
        as_image_dirs: bool = False,
        n_loading_workers: int = 8,
        n_metadata_indexing_workers: int = 8,
        chunk_size: int | Literal["even_split"] = "even_split",
        frame_sorting: None | str = None,
        transform: Callable | None = None,
        buffer_size: int = 32,
        cache_metadata: bool = True,
        use_cached_metadata: bool = True,
        log_interval: int = 1000,
    ):
        r"""
        Args:
            paths (list[Path]): List of to video paths, or directories containing frames
                as individual images.
            batch_size (int): Number of frames to yield per batch.
            as_image_dirs (bool): If True, treat each path as a directory containing
                individual frames. Otherwise, treat it as a video file.
            n_loading_workers (int): Number of worker processes to use for loading and
                decoding frames. -1 means all available cores, -2 means all but 1 core,
                etc. Default is 8.
            n_metadata_indexing_workers (int): Number of worker processes to use for
                indexing video metadata (i.e. number of frames). -1 means all available
                cores, -2 means all but 1 core, etc. Default is 8.
            chunk_size (int | Literal["even_split"]): Number of frames to assign to each
                loading worker at a time. Larger chunks = more efficient decoding, but
                worse load balancing. If "even_split", the maximum meaningful chunk size
                is dynamically determined and used.
            frame_sorting (str | None): When `as_image_dirs` is True, this argument
                specifies how images within each directory should be sorted. If None,
                files are sorted by name. If given as a string, it is used as a regex
                pattern to extract frame numbers from filenames (e.g.
                r"frame\D*(\d+)(?!\d)"). When `as_image_dirs` is False, this argument is
                ignored.
            transform (Callable | None): A function that is to be applied to each frame
                after loading. Note that the following operations are already applied to
                each frame:
                (i) conversion from numpy array to torch tensor,
                (ii) conversion from HWC to CHW format, and
                (iii) conversion from uint8 in [0, 255] to float in [0, 1].
                The transform function, if provided, is applied after these operations.
            buffer_size (int): Number of frames to buffer when reading from video files.
                Buffering is not used when reading from image directories. Larger buffer
                size = more efficient decoding, but more memory usage. Default is 32.
            cache_metadata (bool): Whether to cache video metadata to disk for faster
                loading in the future. Only applies when reading from video files.
            use_cached_metadata (bool): Whether to use cached video metadata if
                available. Only applies when reading from video files.
            log_interval (int): Interval (in number of frames) at which to log progress.
        """
        self.logger = logging.getLogger(__name__)

        # Parse number of workers
        n_loading_workers = self._parse_n_workers(n_loading_workers)
        n_metadata_indexing_workers = self._parse_n_workers(n_metadata_indexing_workers)
        self.n_loading_workers = n_loading_workers
        self.n_metadata_indexing_workers = n_metadata_indexing_workers

        # Store other parameters
        self.batch_size = batch_size
        self.as_image_dirs = as_image_dirs
        self.frame_sorting = frame_sorting
        self.transform = transform
        self.buffer_size = buffer_size
        self.log_interval = log_interval

        # Store canonical Path objects
        self.video_paths = [Path(p) for p in paths]
        self.video_paths_posix = [p.absolute().as_posix() for p in self.video_paths]
        self._check_if_paths_valid()

        # Sort frames if loading from image directories
        if self.as_image_dirs:
            self.sorted_frames_by_video_idx = self._sort_frames_in_image_dirs()
        else:
            self.sorted_frames_by_video_idx = None

        # Index number of frames for each
        self.n_frames_by_video = self._check_n_frames_by_video(
            cache_metadata, use_cached_metadata
        )
        self.n_frames_total = sum(self.n_frames_by_video)

        # Create synchronization primitives for parallel loading
        chunks = self._define_job_chunks(chunk_size)
        self.task_queue = mp.Queue()
        for chunk in chunks:
            self.task_queue.put(chunk)
        for _ in range(self.n_loading_workers):
            self.task_queue.put(None)  # sentinel values to signal worker exit
        # Limit size of decoded_frames_queue to prevent memory bloat. We only need
        # 2*batch_size anyway because when the caller consumes a batch, we can free up
        # that much space for a new batch of decoded frames to be added to the queue (if
        # they're already decoded & pending), while the other half of frames can be
        # immediately consumed by the caller again.
        self.decoded_frames_queue = mp.Queue(maxsize=2 * self.batch_size)
        self.error_queue = mp.Queue()

        # Start worker processes
        self.loading_workers = []
        for worker_id in range(n_loading_workers):
            p = mp.Process(
                target=_worker_process_wrapper, args=(worker_id, self), daemon=True
            )
            p.start()
            self.loading_workers.append(p)

    def _check_if_paths_valid(self):
        # Check if the paths are all valid
        for p in self.video_paths:
            if self.as_image_dirs:
                if not p.is_dir():
                    raise ValueError(
                        f"One of the specified paths {p} is not a valid directory. "
                        "Directories containing individual frame images are expected."
                    )
            else:
                if not p.is_file():
                    raise ValueError(
                        f"One of the specified paths {p} is not a valid file. "
                        "Video files are expected."
                    )

    def _sort_frames_in_image_dirs(self):
        """Sort images if we're loading from directories of images"""
        regex = re.compile(self.frame_sorting) if self.frame_sorting else None
        sorted_frames_by_videoid: list[list[Path]] = []

        # Iterate over the canonical Path objects (self.video_paths) so we
        # consistently store Path keys and avoid relying on caller types
        for video_path in self.video_paths:
            all_files = [f for f in video_path.iterdir() if f.is_file()]
            if regex is None:
                sorting_func = lambda f: f.name
            else:
                sorting_func = lambda f: _extract_frame_number(f.name, regex)
            all_files.sort(key=sorting_func)
            sorted_frames_by_videoid.append(all_files)

        return sorted_frames_by_videoid

    def _parse_n_workers(self, n_workers: int) -> int:
        """Parse the number of workers, handling negative values that indicate
        "all available cores but X" and cap numbers above total CPU count."""
        cpu_count = mp.cpu_count()

        if cpu_count < n_workers:
            self.logger.warning(
                f"Requested {n_workers} workers but only {cpu_count} CPUs available. "
                f"Capping to {cpu_count} workers."
            )
            return cpu_count
        elif 0 < n_workers <= cpu_count:
            return n_workers
        elif n_workers == 0:
            self.logger.warning(
                "Decoding in main process (n_workers=0) not implemented. "
                "Using 1 worker instead (i.e. sequential, but in a different process)."
            )
            return 1
        elif -cpu_count < n_workers < 0:
            # -1 = all cores; -2 = all but 1 core; etc.
            n_workers_effective = cpu_count + 1 + n_workers
            self.logger.info(
                f"Converted n_workers from {n_workers} to {n_workers_effective}."
            )
            return n_workers_effective
        else:
            raise ValueError(f"Invalid n_workers value: {n_workers}")

    def _check_n_frames_by_video(
        self, cache_metadata: bool, use_cached_metadata: bool
    ) -> list[int]:
        """Check how many frames there are in each video. This requires partially
        decoding the video files and it can be quite slow, so we do it in parallel and
        use caches."""
        if self.as_image_dirs:
            return [
                len(sorted_frames) for sorted_frames in self.sorted_frames_by_video_idx
            ]
        else:
            self.logger.info(
                f"Loading metadata for {len(self.video_paths)} videos. "
                "This may take a while if no cached metadata is available."
            )
            metas = Parallel(n_jobs=self.n_metadata_indexing_workers)(
                delayed(get_video_metadata)(path, cache_metadata, use_cached_metadata)
                for path in tqdm(
                    self.video_paths_posix, desc="Checking metadata", disable=None
                )
            )
            return [meta["n_frames"] for meta in metas]

    def _define_job_chunks(self, chunk_size: int | Literal["even_split"]):
        # Even split = split frames evenly among loading workers so that each worker
        # gets a single chunk of approximately equal size. This in principle leads to
        # maximum decoding efficiency, but at the cost of worse load balancing.
        if chunk_size == "even_split":
            chunk_size = int(np.ceil(self.n_frames_total / self.n_loading_workers))

        # Frame spec format:
        # 0th column = video index; 1st column = frame index within that video
        all_frame_specs = np.zeros((self.n_frames_total, 2), dtype="uint32")
        frameid_start_global = 0
        for video_idx, n_frames in enumerate(self.n_frames_by_video):
            frameid_end_global = frameid_start_global + n_frames
            frameids_local = np.arange(n_frames, dtype="uint32")
            all_frame_specs[frameid_start_global:frameid_end_global, 0] = video_idx
            all_frame_specs[frameid_start_global:frameid_end_global, 1] = frameids_local
            frameid_start_global = frameid_end_global
        assert frameid_start_global == self.n_frames_total, "n_frames_total mismatch"

        # Split videos and frames into chunks
        all_chunks = []  # each entry is a vertical slice of all_frame_specs
        n_chunks = int(np.ceil(self.n_frames_total / chunk_size))
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, self.n_frames_total)
            all_chunks.append(all_frame_specs[start_idx:end_idx, :])

        self.logger.info(
            f"Split {self.n_frames_total} total frames to {len(all_chunks)} chunks."
        )
        return all_chunks

    def __iter__(self):
        frames_yielded = 0
        with tqdm(
            total=self.n_frames_total, desc="Decoding frames", disable=None
        ) as pbar:
            while frames_yielded < self.n_frames_total:
                batch_decoded_frames = []
                while (
                    len(batch_decoded_frames) < self.batch_size
                    and frames_yielded < self.n_frames_total
                ):
                    # Check for worker errors
                    try:
                        error = self.error_queue.get_nowait()
                        raise RuntimeError(
                            f"Worker {error['worker_id']} crashed: {error['error']}"
                        )
                    except queue.Empty:
                        pass

                    # Get next decoded frame
                    item = self.decoded_frames_queue.get(block=True)
                    batch_decoded_frames.append(item)
                    frames_yielded += 1

                    if frames_yielded % self.log_interval == 0:
                        self.logger.info(f"Decoded {frames_yielded} frames.")

                pbar.update(len(batch_decoded_frames))

                batch_frame_tensors = torch.stack(
                    [torch.from_numpy(x["frame"]) for x in batch_decoded_frames],
                ).to(torch.float32)
                batch_video_indices = torch.tensor(
                    [x["video_idx"] for x in batch_decoded_frames], dtype=torch.uint32
                )
                batch_frame_indices = torch.tensor(
                    [x["frame_idx"] for x in batch_decoded_frames], dtype=torch.uint32
                )
                yield {
                    "frames": batch_frame_tensors,
                    "video_indices": batch_video_indices,
                    "frame_indices": batch_frame_indices,
                }

        self.logger.info("All frames have been decoded.")

    def __len__(self):
        return int(np.ceil(self.n_frames_total / self.batch_size))

    def __del__(self):
        self.logger.info("Draining decoded frames queue")
        while True:
            try:
                self.decoded_frames_queue.get_nowait()
            except queue.Empty:
                break

        self.logger.info("Joining loading workers")
        for p in self.loading_workers:
            p.join(timeout=1)
            if p.is_alive():
                self.logger.error(
                    f"Loading worker process {p.pid} did not exit cleanly. Terminating."
                )
                p.terminate()
                p.join()
        self.logger.info("All loading workers have been joined.")


def _worker_process_wrapper(worker_id: int, loader: VideoCollectionLoader):
    try:
        _worker_process(worker_id, loader)
    except Exception as e:
        loader.logger.critical(f"Worker {worker_id} encountered an error: {e}")
        loader.error_queue.put({"error": str(e), "worker_id": worker_id})
        raise e


def _worker_process(worker_id: int, loader: VideoCollectionLoader):
    """Worker process that decodes frames and puts them into the decoded_frames_queue"""
    logger = loader.logger
    logger.info(f"Worker {worker_id} started.")

    frame_buffer: dict[tuple[int, int], torch.Tensor] = {}
    curr_decoder = None
    curr_decoded_video_idx = None

    def _get_frame(video_idx: int, frame_idx: int) -> torch.Tensor:
        nonlocal frame_buffer, curr_decoder, curr_decoded_video_idx

        key = (video_idx, frame_idx)
        if key in frame_buffer:
            return frame_buffer[key]

        if curr_decoded_video_idx != video_idx:
            # Switch to a new video
            frame_buffer = {}
            curr_decoder = VideoDecoder(
                loader.video_paths_posix[video_idx],
                seek_mode="exact",
                dimension_order="NCHW",
            )
            curr_decoded_video_idx = video_idx

        # Buffer frames starting from frame_idx
        frame_buffer = {}  # requested data no longer buffered - let's reset the buffer
        decode_until_idx = min(
            frame_idx + loader.buffer_size,
            loader.n_frames_by_video[video_idx],
        )
        frame_indices_to_load = list(range(frame_idx, decode_until_idx))
        frames = curr_decoder.get_frames_at(frame_indices_to_load)  # NCHW tensor
        for i, frameid in enumerate(frame_indices_to_load):
            frame_buffer[(curr_decoded_video_idx, frameid)] = frames.data[i, ...]
        logger.debug(
            f"Worker {worker_id} decoded frames [{frameid}, {decode_until_idx}) "
            f"for video {video_idx}."
        )

        return frame_buffer[key]

    while True:
        chunk = loader.task_queue.get()
        if chunk is None:  # sentinel value to signal exit
            logger.info(f"Worker {worker_id} received exit signal.")
            break

        for i in range(chunk.shape[0]):
            video_idx = chunk[i, 0]
            frameid = chunk[i, 1]

            if loader.as_image_dirs:
                # Load image from directory
                image_path = loader.sorted_frames_by_video_idx[video_idx][frameid]
                frame = imageio.imread(image_path)
                frame = torch.from_numpy(frame)
                if frame.ndim == 2:
                    frame = frame.unsqueeze(-1)  # add channel dim
                frame = frame.permute(2, 0, 1)  # HWC to CHW
            else:
                # Load frame from video
                frame = _get_frame(video_idx, frameid)

            frame = frame.float() / 255.0
            if loader.transform:
                frame = loader.transform(frame)

            data = {
                "frame": frame.cpu().numpy(),  # IPC with tensors is funky - use numpy
                "video_idx": video_idx,
                "frame_idx": frameid,
            }
            loader.decoded_frames_queue.put(data, block=True)

    logger.info(f"Worker {worker_id} exiting.")


def _extract_frame_number(filename: str, regex_pattern: str) -> int:
    matches = re.findall(regex_pattern, filename)
    if len(matches) != 1:
        raise ValueError(
            f"{len(matches)} matches found in filename {filename} "
            f"using regex pattern {regex_pattern}. Only one match is expected."
        )
    try:
        return int(matches[0])
    except ValueError as e:
        raise ValueError(
            f"Failed to parse '{matches[0]}' as int. This substring is extracted "
            f"from filename {filename} using regex pattern {regex_pattern}."
        ) from e
