import numpy as np
from typing import Any


def balance_load_lpt(tasks: dict[Any, int], n_workers: int) -> list[list[Any]]:
    """The Longest Processing Time (LPT) algorithm for load balancing: sort
    tasks by decreasing duration and assigns each task to the worker with
    the currently smallest total assigned load.

    Args:
        tasks (dict[Any, int]): A dict mapping task identifiers (can be
            any hashable type) to their estimated durations.
        n_workers (int): Number of workers to distribute tasks across.

    Returns:
        assignments (list[list[Any]]): A list of lists, where
            assignments[i] contains the IDs of tasks assigned to worker i.
    """
    # Sort tasks by descending duration
    sorted_tasks = sorted(tasks.items(), key=lambda x: x[1], reverse=True)

    # Initialize workers and their current loads
    worker_loads = np.zeros(n_workers)
    assignments = [[] for _ in range(n_workers)]

    # Assign each task to the currently least-loaded worker
    for task_id, duration in sorted_tasks:
        i = int(np.argmin(worker_loads))
        assignments[i].append(task_id)
        worker_loads[i] += duration

    return assignments
