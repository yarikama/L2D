import numpy as np

from l2d.types import JSSPInstance


def _shuffle_rows(two_dim_array: np.ndarray) -> np.ndarray:
    """Independently shuffle each row of a 2D array.

    Args:
        two_dim_array: A 2D array.

    Returns:
        A 2D array with the same shape as the input, but with each row shuffled.
    """
    # Generate random values per element, argsort gives a permutation per row
    return np.take_along_axis(two_dim_array, np.random.random(two_dim_array.shape).argsort(axis=1), axis=1)

def generate_uniform_sjssp_instance(
    num_jobs: int,
    num_machines: int,
    low_bound_processing_time: int,
    high_bound_processing_time: int,
) -> JSSPInstance:
    """
    Generate a random JSSP instance with uniform processing times.

    Returns:
        JSSPInstance: A JSSP instance.
    """
    processing_times = np.random.randint(
        low=low_bound_processing_time,
        high=high_bound_processing_time,
        size=(num_jobs, num_machines),
    )
    machine_assignments_for_each_job = np.tile(np.arange(1, num_machines + 1), (num_jobs, 1))
    shuffled_machine_assignments_for_each_job = _shuffle_rows(machine_assignments_for_each_job)
    return JSSPInstance(times=processing_times, machines=shuffled_machine_assignments_for_each_job)
