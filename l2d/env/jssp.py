"""
Gym environment for the Sequential Job Shop Scheduling Problem (SJSSP).

State representation:

- adj: (num_operations, num_operations) adjacency matrix of the constraint graph
    - self-loops + conjunctive arcs (job order) + disjunctive arcs (machine order)

- features: (num_operations, 2) per-operation features
    - [:, 0] = normalized end-time lower bound
    - [:, 1] = finished mark (1 if scheduled, 0 otherwise)

- omega: (num_jobs,) next schedulable operation ID per job

- mask: (num_jobs,) True if job is fully scheduled

Action: an operation ID (int) from omega where mask is False.

Reward: negative change in makespan lower bound (minimizing makespan).
"""

import gym
import numpy as np

from l2d.config import configs
from l2d.env.end_time_lb import calc_end_time_lower_bound
from l2d.env.get_machine_neighbors import get_machine_neighbors
from l2d.env.left_shift import permissible_left_shift
from l2d.types import (
    AdjMatrix,
    Features,
    MachineAssignments,
    Mask,
    Omega,
    ProcessingTimes,
    ResetResult,
    StepResult,
)


class SJSSP(gym.Env):
    """Sequential Job Shop Scheduling Problem (SJSSP) environment."""

    def __init__(self, num_jobs: int, num_machines: int):
        """
        Initialize the SJSSP environment.

        Args:
            num_jobs: Number of jobs in the instance.
            num_machines: Number of machines in the instance.
        """
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_operations = self.num_jobs * self.num_machines

        # Operation IDs for the first and last operation of each job
        # Job i's operations are [i*n_m, i*n_m+1, ..., i*n_m+(n_m-1)]
        self.first_operation_per_job = np.arange(0, self.num_operations, self.num_machines)
        self.last_operation_per_job = np.arange(self.num_machines - 1, self.num_operations, self.num_machines)

    def is_done(self) -> bool:
        return len(self.schedule) == self.num_operations

    # ── step and its sub-routines ────────────────────────────────────

    def step(self, action: int) -> StepResult:
        """
        Step the environment with the given action.

        Args:
            action: The operation ID to schedule.

        Returns:
            StepResult: The result of the step.
        """
        if action not in self.schedule:
            self._schedule_operation(action)

        features = self._build_features()
        reward = self._compute_reward()

        return StepResult(self.adj, features, reward, self.is_done(), self.omega, self.mask)

    def _schedule_operation(self, action: int) -> None:
        """Execute a single scheduling decision and update all internal state."""
        job_index = action // self.num_machines
        op_index = action % self.num_machines
        self.step_count += 1
        self.finished_mark[job_index, op_index] = 1
        operation_duration = self.durations[job_index, op_index]
        self.schedule.append(action)

        start_time = self._assign_to_machine(action)
        self._advance_candidate(action, job_index)
        self._update_end_time_and_lower_bounds(job_index, op_index, start_time, operation_duration)
        self._update_adjacency(action)

    def _assign_to_machine(self, action: int) -> float:
        """Place the operation on its machine via permissible left shift.

        Returns:
            The scheduled start time.
        """
        start_time, was_inserted = permissible_left_shift(
            action=action,
            durations=self.durations,
            machines=self.machines,
            machine_start_times=self.machine_start_times,
            op_ids_on_machines=self.op_ids_on_machines,
        )
        self.insertion_flags.append(was_inserted)
        self._last_was_inserted = was_inserted
        return start_time

    def _advance_candidate(self, action: int, job_index: int) -> None:
        """Move omega to the next operation of the job, or mark job done."""
        if action not in self.last_operation_per_job:
            self.omega[job_index] += 1
        else:
            self.mask[job_index] = True

    def _update_end_time_and_lower_bounds(
        self, job_index: int, op_index: int, start_time: float, duration: float
    ) -> None:
        """Record end time and recompute all lower bounds."""
        self.end_times[job_index, op_index] = start_time + duration
        self.lower_bounds = calc_end_time_lower_bound(
            self.end_times, self.remaining_durations
        )

    def _update_adjacency(self, action: int) -> None:
        """Update the constraint graph after scheduling *action*."""
        predecessor_op_id, successor_op_id = get_machine_neighbors(
            action, self.op_ids_on_machines
        )

        # Reset row and rebuild edges
        self.adj[action] = 0
        self.adj[action, action] = 1  # self-loop

        # Conjunctive arc: point to same-job predecessor
        if action not in self.first_operation_per_job:
            self.adj[action, action - 1] = 1

        # Disjunctive arcs: same-machine predecessor / successor
        self.adj[action, predecessor_op_id] = 1
        self.adj[successor_op_id, action] = 1

        # If inserted between two existing ops, remove their direct arc
        if self._last_was_inserted and predecessor_op_id != action and successor_op_id != action:
            self.adj[successor_op_id, predecessor_op_id] = 0

    def _build_features(self) -> Features:
        """Concatenate normalized lower bounds and finished marks."""
        return np.concatenate((
            self.lower_bounds.reshape(-1, 1) / configs.et_normalize_coef,
            self.finished_mark.reshape(-1, 1),
        ), axis=1)

    def _compute_reward(self) -> float:
        """Reward = negative makespan increase; small positive if no change."""
        reward = -(self.lower_bounds.max() - self.max_end_time)
        if reward == 0:
            reward = configs.rewardscale
            self.pos_rewards += reward
        self.max_end_time = self.lower_bounds.max()
        return reward

    # ── reset and its sub-routines ───────────────────────────────────

    def reset(self, data: tuple[ProcessingTimes, MachineAssignments]) -> ResetResult:
        """Reset the environment with a new JSSP instance."""
        self._load_instance(data)
        self._init_adjacency_matrix()
        self._init_lower_bounds()
        self._init_candidate_operations()
        self._init_machine_state()

        features = self._build_features()
        return ResetResult(self.adj, features, self.omega, self.mask)

    def _load_instance(self, data: tuple[ProcessingTimes, MachineAssignments]) -> None:
        """Store instance data and reset bookkeeping."""
        self.step_count = 0
        self.machines: MachineAssignments = data[-1]
        self.durations: ProcessingTimes = data[0].astype(np.single)
        self.remaining_durations = np.copy(self.durations)
        self.schedule: list[int] = []
        self.insertion_flags: list[bool] = []
        self._last_was_inserted = False
        self.pos_rewards = 0

    def _init_adjacency_matrix(self) -> None:
        """Build initial adj with self-loops + conjunctive (job-order) arcs."""
        identity = np.eye(self.num_operations, dtype=np.single)
        job_precedence = np.eye(self.num_operations, k=-1, dtype=np.single)
        job_precedence[self.first_operation_per_job] = 0  # first op has no predecessor
        self.adj: AdjMatrix = identity + job_precedence

    def _init_lower_bounds(self) -> None:
        """Compute initial lower bounds (cumulative duration per job)."""
        self.lower_bounds = np.cumsum(self.durations, axis=1, dtype=np.single)
        self.init_quality = self.lower_bounds.max() if not configs.init_quality_flag else 0
        self.max_end_time = self.init_quality
        self.finished_mark = np.zeros_like(self.machines, dtype=np.single)

    def _init_candidate_operations(self) -> None:
        """Set omega to each job's first operation; mask all False."""
        self.omega: Omega = self.first_operation_per_job.astype(np.int64)
        self.mask: Mask = np.full(self.num_jobs, fill_value=False)

    def _init_machine_state(self) -> None:
        """Allocate machine start-time and operation-ID tracking arrays."""
        self.machine_start_times = -configs.high * np.ones_like(
            self.durations.T, dtype=np.int32
        )
        self.op_ids_on_machines = -self.num_jobs * np.ones_like(
            self.durations.T, dtype=np.int32
        )
        self.end_times = np.zeros_like(self.durations, dtype=np.single)

