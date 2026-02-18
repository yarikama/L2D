from pathlib import Path

import numpy as np

from l2d.env.types import JSSPInstance, MachineAssignments, ProcessingTimes


def _shuffle_rows(two_dim_array: np.ndarray) -> np.ndarray:
    """Independently shuffle each row of a 2D array.

    Args:
        two_dim_array: A 2D array.

    Returns:
        A 2D array with the same shape as the input, but with each row shuffled.

    Example:
        >>> arr = np.array([[1, 2, 3],
        ...                 [4, 5, 6]])
        >>> result = _shuffle_rows(arr)
        >>> result  # e.g. array([[3, 1, 2], [5, 4, 6]])
        >>> result.shape == arr.shape  # always True
        True
    """
    # Generate random values per element, argsort gives a permutation per row
    return np.take_along_axis(
        two_dim_array,
        np.random.random(two_dim_array.shape).argsort(axis=1),
        axis=1,
    )


def generate_uniform_times_and_machines_assignment(
    num_jobs: int,
    num_machines: int,
    low_bound_processing_time: int,
    high_bound_processing_time: int,
) -> JSSPInstance:
    """Generate a random JSSP instance with uniform processing times and machines assignments.

    Example:
        >>> instance = generate_uniform_times_and_machines_assignment(2, 3, 1, 10)
        >>> instance.times.shape
        (2, 3)
        >>> instance.machines.shape  # each row is a permutation of [1, 2, 3]
        (2, 3)
    """
    processing_times: ProcessingTimes = np.random.randint(
        low=low_bound_processing_time,
        high=high_bound_processing_time,
        size=(num_jobs, num_machines),
    )
    machine_assignments_for_each_job: MachineAssignments = np.tile(np.arange(1, num_machines + 1), (num_jobs, 1))
    shuffled_machine_assignments_for_each_job: MachineAssignments = _shuffle_rows(machine_assignments_for_each_job)
    return JSSPInstance(times=processing_times, machines=shuffled_machine_assignments_for_each_job)


def generate_uniform_machines_assignment_and_stores_to_file(
    num_jobs: int,
    num_machines: int,
    low_bound_processing_time: int,
    high_bound_processing_time: int,
    file_path: Path = Path('data/generated/'),
    seed: int = 200,
) -> JSSPInstance:
    """Generate a random JSSP instance and store it to a file.

    Example:
        >>> instance = generate_uniform_machines_assignment_and_stores_to_file(
        ...     2, 3, 1, 10, file_path=Path('/tmp'), seed=42,
        ... )
        >>> instance.times.shape
        (2, 3)
        >>> Path('/tmp/generatedData2_3_Seed42.npy').exists()
        True
    """
    instance = generate_uniform_times_and_machines_assignment(
        num_jobs=num_jobs,
        num_machines=num_machines,
        low_bound_processing_time=low_bound_processing_time,
        high_bound_processing_time=high_bound_processing_time,
    )
    np.save(file_path / f'generatedData{num_jobs}_{num_machines}_Seed{seed}.npy', (instance.times, instance.machines))
    return instance
