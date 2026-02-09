"""
Generate random JSSP instances.

A JSSP instance is a tuple of (times, machines):
- times[i, j]: processing time of job i's j-th operation
- machines[i, j]: which machine (1-indexed) job i's j-th operation runs on

Each job visits every machine exactly once (in a random order).
"""

import numpy as np


def _shuffle_rows(two_dim_array: np.ndarray) -> np.ndarray:
    """Independently shuffle each row of a 2D array.

    Args:
        two_dim_array: A 2D array.

    Returns:
        A 2D array with the same shape as the input, but with each row shuffled.
    """
    # Generate random values per element, argsort gives a permutation per row
    return np.take_along_axis(two_dim_array, np.random.random(two_dim_array.shape).argsort(axis=1), axis=1)


def generate_uniform_instance(
    n_j: int,
    n_m: int,
    low: int,
    high: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a random JSSP instance with uniform processing times.

    Args:
        n_j: Number of jobs.
        n_m: Number of machines.
        low: Lower bound of processing times (inclusive).
        high: Upper bound of processing times (exclusive).

    Returns:
        times: (n_j, n_m) array of processing times.
        machines: (n_j, n_m) array of machine assignments (1-indexed).
    """
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    
    # Each job visits machines [1, 2, ..., n_m] in a shuffled order
    machines = np.tile(np.arange(1, n_m + 1), (n_j, 1))
    machines = _shuffle_rows(machines)
    
    return times, machines
