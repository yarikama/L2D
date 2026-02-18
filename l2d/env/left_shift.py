"""Permissible left-shift scheduling for the JSSP environment.

When an operation is assigned to a machine, this module finds the earliest
feasible start time by:

1. Computing when the operation is ready (both its job-predecessor and
   machine must be free).
2. Scanning for idle gaps between already-scheduled operations on that
   machine where the new operation could fit ("insertion").
3. If no gap is large enough, appending to the end of the machine's
   schedule.

The key entry point is `permissible_left_shift()`.
"""

import numpy as np

from l2d.config import configs

# ── Public API ───────────────────────────────────────────────────────


def permissible_left_shift(
    action: int,
    durations: np.ndarray,
    machines: np.ndarray,
    machine_start_times: np.ndarray,
    op_ids_on_machines: np.ndarray,
) -> tuple[float, bool]:
    """Schedule an operation at the earliest feasible time on its machine.

    Attempts to insert the operation into an idle gap between existing
    operations.  If no gap is large enough, appends it after the last
    scheduled operation.

    Args:
        action: Flat operation ID to schedule.
        durations: (num_jobs, num_machines) processing-time matrix.
        machines: (num_jobs, num_machines) machine-assignment matrix
            (1-indexed).
        machine_start_times: (num_machines, num_jobs) start times of
            operations already placed on each machine.  Empty slots hold
            ``-configs.high``.  **Modified in-place.**
        op_ids_on_machines: (num_machines, num_jobs) operation IDs placed
            on each machine.  Empty slots are negative.  **Modified
            in-place.**

    Returns:
        start_time: The scheduled start time of the operation.
        was_inserted: True if the operation was inserted into a gap
            (rather than appended at the end).
    """
    job_ready_time, machine_ready_time = _calc_ready_times(
        action, durations, machines, machine_start_times, op_ids_on_machines,
    )

    operation_duration = np.take(durations, action)
    machine_index = np.take(machines, action) - 1  # 1-indexed → 0-indexed

    # Views into this machine's schedule (modified in-place by helpers)
    mch_start_times = machine_start_times[machine_index]
    mch_op_ids = op_ids_on_machines[machine_index]

    # Positions where the job is already ready before the next scheduled op
    candidate_positions = np.where(job_ready_time < mch_start_times)[0]

    if len(candidate_positions) == 0:
        # No scheduled op starts after job_ready_time → append at end
        start_time = _append_at_end(
            action, job_ready_time, machine_ready_time,
            mch_start_times, mch_op_ids,
        )
        return start_time, False

    # Check which candidate gaps are wide enough for the operation
    gap_indices, gap_positions, gap_earliest_starts = _find_insertion_gaps(
        operation_duration, job_ready_time, durations,
        candidate_positions, mch_start_times, mch_op_ids,
    )

    if len(gap_positions) == 0:
        # Gaps exist but none is wide enough → append at end
        start_time = _append_at_end(
            action, job_ready_time, machine_ready_time,
            mch_start_times, mch_op_ids,
        )
        return start_time, False

    # Insert into the earliest valid gap
    start_time = _insert_into_gap(
        action, gap_indices, gap_positions, gap_earliest_starts,
        mch_start_times, mch_op_ids,
    )
    return start_time, True


# ── Ready-time computation ───────────────────────────────────────────


def _calc_ready_times(
    action: int,
    durations: np.ndarray,
    machines: np.ndarray,
    machine_start_times: np.ndarray,
    op_ids_on_machines: np.ndarray,
) -> tuple[float, float]:
    """Compute the earliest time *action* can start due to job and machine constraints.

    Returns:
        job_ready_time: When the preceding operation of the same job finishes.
            0 if *action* is the first operation of its job.
        machine_ready_time: When the machine last becomes free (end time of
            the last scheduled operation on that machine).
            0 if no operation has been scheduled on the machine yet.
    """
    num_machines = machines.shape[1]
    machine_index = np.take(machines, action) - 1

    # ── Job ready time ───────────────────────────────────────────────
    is_first_op_of_job = (action % num_machines == 0)

    if is_first_op_of_job:
        job_ready_time: float = 0
    else:
        job_predecessor = action - 1
        predecessor_duration = np.take(durations, job_predecessor)
        predecessor_machine = np.take(machines, job_predecessor) - 1

        # Find where the predecessor is placed on its machine
        predecessor_slot = np.where(
            op_ids_on_machines[predecessor_machine] == job_predecessor
        )
        predecessor_start = machine_start_times[predecessor_machine][predecessor_slot]
        job_ready_time = (predecessor_start + predecessor_duration).item()

    # ── Machine ready time ───────────────────────────────────────────
    scheduled_slots = np.where(op_ids_on_machines[machine_index] >= 0)[0]

    if len(scheduled_slots) == 0:
        machine_ready_time: float = 0
    else:
        last_op_on_machine = op_ids_on_machines[machine_index][scheduled_slots[-1]]
        last_op_duration = np.take(durations, last_op_on_machine)
        last_op_start = machine_start_times[machine_index][np.where(
            machine_start_times[machine_index] >= 0
        )][-1]
        machine_ready_time = (last_op_start + last_op_duration).item()

    return job_ready_time, machine_ready_time


# ── Placement strategies ─────────────────────────────────────────────


def _append_at_end(
    action: int,
    job_ready_time: float,
    machine_ready_time: float,
    mch_start_times: np.ndarray,
    mch_op_ids: np.ndarray,
) -> float:
    """Place *action* after all existing operations on the machine.

    Finds the first empty slot (marked by ``-configs.high``) and writes
    the start time and operation ID there.
    """
    empty_slot = np.where(mch_start_times == -configs.high)[0][0]
    start_time = max(job_ready_time, machine_ready_time)
    mch_start_times[empty_slot] = start_time
    mch_op_ids[empty_slot] = action
    return start_time


def _find_insertion_gaps(
    operation_duration: float,
    job_ready_time: float,
    durations: np.ndarray,
    candidate_positions: np.ndarray,
    mch_start_times: np.ndarray,
    mch_op_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify gaps between scheduled operations that can fit *operation_duration*.

    A "gap" is the idle time between the end of one operation and the
    start of the next on the same machine.

    Returns:
        gap_indices: Indices into *candidate_positions* where gaps are large enough.
        gap_positions: The actual slot positions in the machine schedule.
        gap_earliest_starts: The earliest possible start time at each candidate position.
    """
    candidate_start_times = mch_start_times[candidate_positions]
    candidate_durations = np.take(durations, mch_op_ids[candidate_positions])

    # Earliest start at the first candidate = max(job_ready, end of preceding op)
    preceding_slot = candidate_positions[0] - 1
    preceding_end = mch_start_times[preceding_slot] + np.take(durations, [mch_op_ids[preceding_slot]])
    earliest_start_at_first = max(job_ready_time, preceding_end)

    # For remaining candidates, earliest start = end of the candidate before
    gap_earliest_starts = np.append(
        earliest_start_at_first,
        (candidate_start_times + candidate_durations)[:-1],
    )

    # Gap = time between start of next op and earliest we could start
    available_gaps = candidate_start_times - gap_earliest_starts
    gap_indices = np.where(operation_duration <= available_gaps)[0]
    gap_positions = np.take(candidate_positions, gap_indices)

    return gap_indices, gap_positions, gap_earliest_starts


def _insert_into_gap(
    action: int,
    gap_indices: np.ndarray,
    gap_positions: np.ndarray,
    gap_earliest_starts: np.ndarray,
    mch_start_times: np.ndarray,
    mch_op_ids: np.ndarray,
) -> float:
    """Insert *action* into the earliest valid gap on the machine.

    Shifts subsequent entries to make room for the new operation.
    """
    earliest_gap_idx = gap_indices[0]
    insert_position = gap_positions[0]
    start_time = gap_earliest_starts[earliest_gap_idx]

    # Shift arrays right at insert_position and write new entry
    mch_start_times[:] = np.insert(mch_start_times, insert_position, start_time)[:-1]
    mch_op_ids[:] = np.insert(mch_op_ids, insert_position, action)[:-1]

    return start_time


# ── Test harness ─────────────────────────────────────────────────────


if __name__ == "__main__":
    import time

    from l2d.env.jssp import SJSSP
    from l2d.env.uni_instance_gen import generate_uniform_times_and_machines_assignment

    n_j = 3
    n_m = 3
    low = 1
    high = 99
    SEED = 10
    np.random.seed(SEED)
    env = SJSSP(num_jobs=n_j, num_machines=n_m)

    t1 = time.time()
    data = generate_uniform_times_and_machines_assignment(
        num_jobs=n_j,
        num_machines=n_m,
        low_bound_processing_time=low,
        high_bound_processing_time=high,
    )
    print('Dur')
    print(data.times)
    print('Mach')
    print(data.machines)
    print()

    machine_start_times = -configs.high * np.ones_like(data.times.transpose(), dtype=np.int32)
    op_ids_on_machines = -n_j * np.ones_like(data.times.transpose(), dtype=np.int32)

    reset_result = env.reset(data)
    omega, mask = reset_result.omega, reset_result.mask
    rewards = []
    flags = []
    while True:
        action = np.random.choice(omega[np.where(mask == 0)])
        print(action)

        step_result = env.step(action)
        reward, omega, mask = step_result.reward, step_result.omega, step_result.mask

        start_time, was_inserted = permissible_left_shift(
            action=action,
            durations=data.times.astype(np.single),
            machines=data.machines,
            machine_start_times=machine_start_times,
            op_ids_on_machines=op_ids_on_machines,
        )
        flags.append(was_inserted)
        print('op_ids_on_machines\n', op_ids_on_machines)
        rewards.append(reward)
        print()
        if env.is_done():
            break

    t2 = time.time()
    print(f'elapsed: {t2 - t1:.4f}s')
    print('machine_start_times\n', machine_start_times)
    print('env.op_ids_on_machines\n', env.op_ids_on_machines)
