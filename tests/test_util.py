from pvio.util import balance_load_lpt


def test_balance_load_lpt_simple():
    tasks = {"a": 5, "b": 4, "c": 3, "d": 2}
    assignments = balance_load_lpt(tasks, 2)
    assert len(assignments) == 2
    assigned = [t for worker in assignments for t in worker]
    assert set(assigned) == set(tasks.keys())


def test_balance_load_lpt_single_worker():
    tasks = {"a": 5, "b": 1}
    assignments = balance_load_lpt(tasks, 1)
    assert len(assignments) == 1
    assert set(assignments[0]) == set(tasks.keys())


def test_balance_load_lpt_more_workers_than_tasks():
    tasks = {"a": 2}
    assignments = balance_load_lpt(tasks, 4)
    # There should be 4 worker lists, and one of them should contain the task
    assert len(assignments) == 4
    flattened = [t for worker in assignments for t in worker]
    assert flattened.count("a") == 1
