"""Asynchronous H5 file writer using multiprocessing to offload H5 write operations
to a separate process.

Note: there is no true parallelism in that H5 write is NOT parallelized - there is
simply a separate process that handles all H5 write operations sequentially. Still, this
frees the main process up for things like processing the next batch of data while IO is
going on in the background.

Example usage:

    h5_write_manager = H5WriteManager()

    with h5_write_manager.File("myfile.h5", "w") as f:
        grp = f.create_group("my_group")
        dset = grp.create_dataset("my_dataset", shape=(1000, 128, 128), dtype="float32")

        for batch in dataloader:
            batch_input = batch["frames"]
            frameids = batch["frame_indices"]

            # Process batch_input to get batch_output
            batch_output = my_model(batch_input)

            # Write batch_output to dataset asynchronously
            future = dset.partial_write(np.s_[frameids, :, :], batch_output.numpy())

            # The function above returns immediately with a Future object that can be
            # used to see if the write is done. In the meantime, you can proceed to the
            # next batch in the main process without waiting for the write to complete

            # (Optional) If you **do** want to wait for the write to complete:
            future.get_result()  # this blocks until the write is done

"""

try:
    import h5py
except ImportError as e:
    raise ImportError(
        "h5py is required for pvio.h5_writer but is not installed. Please install "
        'with h5py support using `pip install "parallel-video-io[h5_writer]"` or '
        '`pip install -e ".[h5_writer]"`.'
    ) from e

import numpy as np
from time import perf_counter
from typing import Any
from multiprocessing import Queue, Process
from queue import Empty
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from abc import ABC
from uuid import uuid4
from collections import defaultdict
import logging


logger = logging.getLogger(__name__)
_MODULE_SENTINEL = object()


class CommandType(Enum):
    OPEN_FILE = "OPEN_FILE"
    CREATE_GROUP = "CREATE_GROUP"
    CREATE_DATASET = "CREATE_DATASET"
    CREATE_ATTR = "CREATE_ATTR"
    DATASET_WRITE = "DATASET_WRITE"
    FLUSH = "FLUSH"
    CLOSE_FILE = "CLOSE_FILE"
    QUIT = "QUIT"


@dataclass
class Command:
    command_type: CommandType
    caller_uuid: str
    child_uuid: str | None = None
    child_attrs_uuid: str | None = None
    args: tuple[Any, ...] | None = None
    kwargs: dict[str, Any] | None = None
    data: np.ndarray | None = None
    slice_: Any | None = None


@dataclass
class RemoteResponse:
    success: bool
    data: Any | None = None
    error: Exception | None = None
    walltime: float | None = None


class Future:
    """A simple Future implementation using Queue to get results from async writes"""

    def __init__(self):
        self._queue: Queue = Queue(maxsize=1)

    def set_result(self, result: Any) -> None:
        self._queue.put(result)

    def get_result(self, timeout: float | None = None) -> RemoteResponse | None:
        try:
            response = self._queue.get(timeout=timeout)
            return response
        except Empty:
            return None


class AsyncH5Object(ABC):
    def __init__(self, workload_queue: Queue):
        self._uuid = str(uuid4())
        self._workload_queue = workload_queue
        if not isinstance(self, AsyncAttributeManager):
            self.attrs = AsyncAttributeManager(_MODULE_SENTINEL, workload_queue)


class AsyncAttributeManager(AsyncH5Object):
    def __init__(self, _token, *args, **kwargs):
        if _token is not _MODULE_SENTINEL:
            raise RuntimeError(
                "Set attributes via AsyncFile.attrs, AsyncGroup.attrs, or "
                "AsyncDataset.attrs. Do not call AsyncAttributeManager() directly."
            )
        super().__init__(*args, **kwargs)

    def create(self, name, data, *args, **kwargs):
        command = Command(
            command_type=CommandType.CREATE_ATTR,
            caller_uuid=self._uuid,
            args=(name, data) + args,
            kwargs=kwargs,
        )
        future = Future()
        self._workload_queue.put((command, future))
        response: RemoteResponse = future.get_result()
        if response.success:
            logger.debug(f"Successfully created attribute {name} remotely")
        else:
            logger.error(
                f"Failed to create attribute {name} remotely: "
                f"Error from worker process: {response.error}"
            )
            raise response.error

    def __setitem__(self, name: str, val: Any):
        return self.create(name, val)


class AsyncGroup(AsyncH5Object):
    def __init__(self, _token, *args, **kwargs):
        if _token is not _MODULE_SENTINEL:
            raise RuntimeError(
                "To create new groups, call .create_group() on AsyncFile or AsyncGroup "
                "objects. Do not call AsyncGroup() directly."
            )
        super().__init__(*args, **kwargs)

    def create_group(self, name: str, *args, **kwargs):
        new_group = AsyncGroup(_MODULE_SENTINEL, self._workload_queue)
        command = Command(
            command_type=CommandType.CREATE_GROUP,
            caller_uuid=self._uuid,
            child_uuid=new_group._uuid,
            child_attrs_uuid=new_group.attrs._uuid,
            args=(name,) + args,
            kwargs=kwargs,
        )
        future: Future = Future()
        self._workload_queue.put((command, future))
        response: RemoteResponse = future.get_result()
        if response.success:
            logger.debug(f"Successfully created group {name} remotely")
            return new_group
        else:
            logger.error(
                f"Failed to create group {name} remotely: "
                f"Error from worker process: {response.error}"
            )
            raise response.error

    def create_dataset(self, name: str, *args, **kwargs):
        new_dataset = AsyncDataset(_MODULE_SENTINEL, self._workload_queue)
        command = Command(
            command_type=CommandType.CREATE_DATASET,
            caller_uuid=self._uuid,
            child_uuid=new_dataset._uuid,
            child_attrs_uuid=new_dataset.attrs._uuid,
            args=(name,) + args,
            kwargs=kwargs,
        )
        future: Future = Future()
        self._workload_queue.put((command, future))
        response: RemoteResponse = future.get_result()
        if response.success:
            logger.debug(f"Successfully created dataset {name} remotely")
            return new_dataset
        else:
            logger.error(
                f"Failed to create dataset {name} remotely: "
                f"Error from worker process: {response.error}"
            )
            raise response.error


class AsyncFile(AsyncGroup):
    def __init__(self, _token, *args, **kwargs):
        super().__init__(_token, *args, **kwargs)  # let AsyncGroup handle checks

    def close(self):
        command = Command(
            command_type=CommandType.CLOSE_FILE,
            caller_uuid=self._uuid,
        )
        future: Future = Future()
        self._workload_queue.put((command, future))
        response: RemoteResponse = future.get_result()
        if response.success:
            logger.debug(f"Successfully closed H5 file remotely")
        else:
            logger.error(
                f"Failed to close H5 file remotely: "
                f"Error from worker process: {response.error}"
            )
            raise response.error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class AsyncDataset(AsyncH5Object):
    def __init__(self, _token, *args, **kwargs):
        if _token is not _MODULE_SENTINEL:
            raise RuntimeError(
                "To create new datasets, call .create_dataset() on AsyncFile or "
                "AsyncGroup objects. Do not call AsyncDataset() directly."
            )
        super().__init__(*args, **kwargs)

    def partial_write(self, slice_, val):
        """Use this instead of __setitem__ to write to a dataset asynchronously and get
        a Future to track the write operation. Use ``np.s_[...]`` to create a
        numpy-style slice object.

        Example:
            Instead of
                >>> dataset[:100, :] = my_array
            use
                >>> future = dataset.partial_write(np.s_[:100, :], my_array)

            Then, you can call
                >>> is_done: bool = future.done()
            to check if the write is complete, or
                >>> response: RemoteResponse = future.get_result()
            to wait for the write to complete and get the response from the worker
            process. You can also use a timeout:
                >>> response: RemoteResponse | None = future.get_result(timeout=1)
            to wait up to 1 second for the write to complete. If the timeout is reached,
            None is returned.
        """
        command = Command(
            command_type=CommandType.DATASET_WRITE,
            caller_uuid=self._uuid,
            slice_=slice_,
            data=val,
        )
        future: Future = Future()
        self._workload_queue.put((command, future))
        return future


class H5WriteManager:
    def __init__(self, max_queue_size: int = 100):
        """Create an H5 writer pool with a separate process to handle H5 write
        operations.

        Args:
            max_queue_size: Maximum number of pending H5 write operations in the queue.
                When the queue is full, attempts to add new write operations will block.
        """
        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._process: Process = Process(
            target=_writer_process_loop, args=(self._queue,), daemon=True
        )
        self._process.start()

    def File(self, name: str | Path, mode: str = "r", *args, **kwargs) -> AsyncFile:
        """Use this to replace direct h5py.File construction.

        Example:
            Instead of
                >>> f = h5py.File("myfile.h5", "w")
            use
                >>> h5_write_manager = H5WriteManager()
                >>> f = h5_write_manager.File("myfile.h5", "w")

            Context manager also works:
                >>> with h5_write_manager.File("myfile.h5", "w") as f:
                ...     # do stuff with f
        """
        if mode == "r":
            raise RuntimeError(
                "H5WriteManager is meant to manage write operations. Reading from H5 "
                "files is not intended and not supported."
            )

        file = AsyncFile(_MODULE_SENTINEL, self._queue)
        file.filename = name
        file.mode = mode

        command = Command(
            command_type=CommandType.OPEN_FILE,
            caller_uuid="root_manager",
            child_uuid=file._uuid,
            child_attrs_uuid=file.attrs._uuid,
            args=(str(name), mode) + args,
            kwargs=kwargs,
        )
        future: Future = Future()
        self._queue.put((command, future))
        response: RemoteResponse = future.get_result()
        if response.success:
            logger.info(f"Successfully opened H5 file {name} remotely")
            return file
        else:
            logger.error(
                f"Failed to open H5 file {name} remotely: "
                f"Error from worker process: {response.error}"
            )
            raise response.error

    def flush(self, timeout: float | None = None) -> None:
        """Block until all pending H5 write operations are complete. If timeout is None,
        wait indefinitely. Otherwise, wait up to `timeout` seconds."""
        command = Command(command_type=CommandType.FLUSH, caller_uuid="root_manager")
        future = Future()
        self._queue.put((command, future))
        response: RemoteResponse | None = future.get_result(timeout=timeout)
        if response is None:
            raise TimeoutError("Hit timeout while waiting for H5 writes to complete")
        if response.success:
            logger.info("All pending H5 write operations complete")
        else:
            logger.error(
                f"Error while flushing H5 writes: Error from worker process: "
                f"{response.error}"
            )
            raise response.error

    def shutdown(self, force: bool = False) -> None:
        """Shut down the H5 writer process. Call this when you are done with all H5
        write operations. If `force` is True, terminate the process immediately without
        waiting for pending operations to complete."""
        if not force:
            self.flush()

        command = Command(
            command_type=CommandType.QUIT,
            caller_uuid="root_manager",
        )
        future = Future()
        self._queue.put((command, future))  # not actually expecting a response
        self._process.join()

    def __del__(self):
        self.shutdown(force=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(force=False)


def _writer_process_loop(workload_queue: Queue) -> None:
    obj_lookup: dict[str, Any] = {}
    children_lookup: dict[str, list[str]] = defaultdict(list)

    while True:
        try:
            command, future = workload_queue.get(timeout=0.1)
            assert isinstance(command, Command)
            assert isinstance(future, Future)
        except Empty:
            continue

        start_time = perf_counter()
        try:
            if command.command_type == CommandType.QUIT:
                logger.info("Received QUIT command, breaking writer loop")
                response = RemoteResponse(success=True, data=None)
                response.walltime = perf_counter() - start_time
                future.set_result(response)
                break
            if command.command_type == CommandType.OPEN_FILE:
                logger.debug("Handling OPEN_FILE command")
                response = _handle_open_file(command, obj_lookup, children_lookup)
            elif command.command_type == CommandType.CREATE_GROUP:
                logger.debug("Handling CREATE_GROUP command")
                response = _handle_create_group(command, obj_lookup, children_lookup)
            elif command.command_type == CommandType.CREATE_DATASET:
                logger.debug("Handling CREATE_DATASET command")
                response = _handle_create_dataset(command, obj_lookup, children_lookup)
            elif command.command_type == CommandType.CREATE_ATTR:
                logger.debug("Handling CREATE_ATTR command")
                response = _handle_create_attr(command, obj_lookup)
            elif command.command_type == CommandType.DATASET_WRITE:
                logger.debug("Handling DATASET_WRITE command")
                response = _handle_dataset_partial_write(command, obj_lookup)
            elif command.command_type == CommandType.FLUSH:
                logger.debug("Handling FLUSH command")
                response = _handle_flush(command, obj_lookup)
            elif command.command_type == CommandType.CLOSE_FILE:
                logger.debug("Handling CLOSE_FILE command")
                response = _handle_close_file(command, obj_lookup, children_lookup)
            else:
                raise ValueError(f"Unknown command type: {command.command_type}")

        except Exception as e:
            logger.error(f"Exception in H5 writer process: {e}")
            response = RemoteResponse(success=False, error=e)

        response.walltime = perf_counter() - start_time
        future.set_result(response)

    logger.info("H5 writer process exiting")


def _handle_open_file(
    command: Command,
    obj_lookup: dict[str, AsyncH5Object],
    children_lookup: dict[str, list[str]],
) -> RemoteResponse:
    file = h5py.File(*command.args, **(command.kwargs or {}))
    obj_lookup[command.child_uuid] = file
    obj_lookup[command.child_attrs_uuid] = file.attrs
    children_lookup[command.child_uuid].append(command.child_attrs_uuid)
    return RemoteResponse(success=True, data=None)


def _handle_create_group(
    command: Command,
    obj_lookup: dict[str, AsyncH5Object],
    children_lookup: dict[str, list[str]],
) -> RemoteResponse:
    # Note: this also works for File since File inherits from Group
    caller_group: h5py.Group = obj_lookup[command.caller_uuid]
    new_group = caller_group.create_group(*command.args, **(command.kwargs or {}))
    obj_lookup[command.child_uuid] = new_group
    obj_lookup[command.child_attrs_uuid] = new_group.attrs
    children_lookup[command.caller_uuid].append(command.child_uuid)
    children_lookup[command.child_uuid].append(command.child_attrs_uuid)
    return RemoteResponse(success=True, data=None)


def _handle_create_dataset(
    command: Command,
    obj_lookup: dict[str, AsyncH5Object],
    children_lookup: dict[str, list[str]],
) -> RemoteResponse:
    # Note: this also works for File since File inherits from Group
    caller_group: h5py.Group = obj_lookup[command.caller_uuid]
    new_dataset = caller_group.create_dataset(*command.args, **(command.kwargs or {}))
    obj_lookup[command.child_uuid] = new_dataset
    obj_lookup[command.child_attrs_uuid] = new_dataset.attrs
    children_lookup[command.caller_uuid].append(command.child_uuid)
    children_lookup[command.child_uuid].append(command.child_attrs_uuid)
    return RemoteResponse(success=True, data=None)


def _handle_create_attr(
    command: Command, obj_lookup: dict[str, AsyncH5Object]
) -> RemoteResponse:
    attrs: h5py.AttributeManager = obj_lookup[command.caller_uuid]
    attrs.create(*command.args, **(command.kwargs or {}))
    return RemoteResponse(success=True, data=None)


def _handle_dataset_partial_write(
    command: Command, obj_lookup: dict[str, AsyncH5Object]
) -> RemoteResponse:
    dataset: h5py.Dataset = obj_lookup[command.caller_uuid]
    dataset[command.slice_] = command.data
    return RemoteResponse(success=True, data=None)


def _handle_flush(
    command: Command, obj_lookup: dict[str, AsyncH5Object]
) -> RemoteResponse:
    for obj in obj_lookup.values():
        if isinstance(obj, h5py.File):
            obj.flush()
    return RemoteResponse(success=True, data=None)


def _handle_close_file(
    command: Command,
    obj_lookup: dict[str, AsyncH5Object],
    children_lookup: dict[str, list[str]],
) -> RemoteResponse:
    file: h5py.File = obj_lookup[command.caller_uuid]
    file.close()

    # Clean up all child objects recursively
    to_delete = [command.caller_uuid]
    while to_delete:
        current = to_delete.pop()
        to_delete.extend(children_lookup[current])
        del obj_lookup[current]
        del children_lookup[current]
    return RemoteResponse(success=True, data=None)
