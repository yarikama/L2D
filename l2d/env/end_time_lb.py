"""End-time lower bound calculation for JSSP operations.

For each operation, compute a lower bound on its completion time by summing
the remaining (unscheduled) durations along the job's processing chain.

WARNING: `calc_end_time_lower_bound` mutates `remaining_durations` in place
for performance. The caller (SJSSP.step) passes a dedicated copy (`dur_cp`).
"""

import numpy as np


def _last_nonzero_indices(arr: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the (row, col) indices of the last nonzero element along `axis` for each row.

    Rows that are entirely zero are excluded from the result.
    """
    mask = arr != 0
    last_pos = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    has_nonzero = mask.any(axis=axis)
    rows = np.arange(arr.shape[0])[has_nonzero]
    cols = last_pos[has_nonzero]
    return rows, cols


def calc_end_time_lower_bound(
    end_times: np.ndarray, remaining_durations: np.ndarray
) -> np.ndarray:
    """
    Compute end-time lower bounds for all operations.

    For scheduled operations (nonzero in end_times), the LB is their actual end time
    plus the cumulative duration of subsequent unscheduled operations in the same job.

    NOTE: This function mutates `remaining_durations` in place.

    Args:
        end_times: (n_j, n_m) array where nonzero values are actual end times
            of already-scheduled operations.
        remaining_durations: (n_j, n_m) array of original durations, will be
            zeroed out at scheduled positions.

    Returns:
        (n_j, n_m) array of end-time lower bounds.
    """
    rows, cols = _last_nonzero_indices(end_times, axis=1)

    # Zero out durations of already-scheduled operations
    remaining_durations[end_times != 0] = 0
    # Restore the last scheduled operation's actual end time as the chain start
    remaining_durations[rows, cols] = end_times[rows, cols]

    # Cumulative sum gives the lower bound chain
    cumulative = np.cumsum(remaining_durations, axis=1)
    # Only keep LBs for unscheduled positions
    cumulative[end_times != 0] = 0

    return end_times + cumulative
