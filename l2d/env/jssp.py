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
from l2d.env.left_shift import permissibleLeftShift
from l2d.env.get_machine_neighbors import get_machine_neighbors
from l2d.types import (
    StepResult,
    ResetResult,
    AdjMatrix,
    Features,
    Omega,
    Mask,
    ProcessingTimes,
    MachineAssignments,
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

    def step(self, action: int) -> StepResult:
        """
        Step the environment with the given action.

        Args:
            action: The operation ID to schedule.

        Returns:
            adj: The adjacency matrix of the constraint graph.
            features: The features of the operations.
            reward: The reward for the action.
            done: Whether the environment is done.
            omega: The next schedulable operation ID per job.
            mask: True if job is fully scheduled.
        """
        # Redundant action (already scheduled) has no effect
        if action not in self.schedule:
            job = action // self.num_machines
            op_idx = action % self.num_machines
            self.step_count += 1
            self.finished_mark[job, op_idx] = 1
            dur_a = self.durations[job, op_idx]
            self.schedule.append(action)

            # Schedule via permissible left shift
            start_time, inserted = permissibleLeftShift(
                a=action, durMat=self.durations, mchMat=self.machines,
                mchsStartTimes=self.machine_start_times,
                opIDsOnMchs=self.op_ids_on_machines,
            )
            self.insertion_flags.append(inserted)

            # Update candidate operations (omega) and completion mask
            if action not in self.last_operation_per_job:
                self.omega[job] += 1
            else:
                self.mask[job] = True

            # Record end time and recompute lower bounds
            self.end_times[job, op_idx] = start_time + dur_a
            self.lower_bounds = calc_end_time_lower_bound(
                self.end_times, self.remaining_durations
            )

            # Update adjacency matrix (constraint graph)
            predecessor, successor = get_machine_neighbors(
                action, self.op_ids_on_machines
            )
            self.adj[action] = 0
            self.adj[action, action] = 1  # self-loop
            if action not in self.first_operation_per_job:
                self.adj[action, action - 1] = 1  # conjunctive arc (job order)
            self.adj[action, predecessor] = 1  # disjunctive arc (machine order)
            self.adj[successor, action] = 1
            # Remove old arc when inserting between two operations
            if inserted and predecessor != action and successor != action:
                self.adj[successor, predecessor] = 0

        # Build features
        features = np.concatenate((
            self.lower_bounds.reshape(-1, 1) / configs.et_normalize_coef,
            self.finished_mark.reshape(-1, 1),
        ), axis=1)

        # Reward = negative makespan increase
        reward = -(self.lower_bounds.max() - self.max_end_time)
        if reward == 0:
            reward = configs.rewardscale
            self.pos_rewards += reward
        self.max_end_time = self.lower_bounds.max()

        return StepResult(self.adj, features, reward, self.is_done(), self.omega, self.mask)

    def reset(self, data: tuple[ProcessingTimes, MachineAssignments]) -> ResetResult:
        self.step_count = 0
        self.machines: MachineAssignments = data[-1]
        self.durations: ProcessingTimes = data[0].astype(np.single)
        self.remaining_durations = np.copy(self.durations)
        self.schedule = []
        self.insertion_flags = []
        self.pos_rewards = 0

        # Adjacency matrix: self-loops + conjunctive (job-order) arcs
        # Each operation points to its predecessor in the same job
        identity = np.eye(self.num_operations, dtype=np.single)
        job_precedence = np.eye(self.num_operations, k=-1, dtype=np.single)
        job_precedence[self.first_operation_per_job] = 0  # first op has no predecessor
        self.adj: AdjMatrix = identity + job_precedence

        # Initial lower bounds = cumulative duration along each job
        self.lower_bounds = np.cumsum(self.durations, axis=1, dtype=np.single)
        self.initQuality = self.lower_bounds.max() if not configs.init_quality_flag else 0
        self.max_end_time = self.initQuality
        self.finished_mark = np.zeros_like(self.machines, dtype=np.single)

        features: Features = np.concatenate((
            self.lower_bounds.reshape(-1, 1) / configs.et_normalize_coef,
            self.finished_mark.reshape(-1, 1),
        ), axis=1)

        # Candidate operations: each job starts with its first operation
        self.omega: Omega = self.first_operation_per_job.astype(np.int64)
        self.mask: Mask = np.full(self.num_jobs, fill_value=False)

        # Machine state tracking (transposed: n_m x n_j)
        self.machine_start_times = -configs.high * np.ones_like(
            self.durations.T, dtype=np.int32
        )
        self.op_ids_on_machines = -self.num_jobs * np.ones_like(
            self.durations.T, dtype=np.int32
        )

        # End times of scheduled operations (filled in during step)
        self.end_times = np.zeros_like(self.durations, dtype=np.single)

        return ResetResult(self.adj, features, self.omega, self.mask)

    # ── Backward compatibility aliases ──────────────────────────────
    @property
    def number_of_jobs(self):
        return self.num_jobs

    @property
    def number_of_machines(self):
        return self.num_machines

    @property
    def number_of_tasks(self):
        return self.num_operations

    @property
    def posRewards(self):
        return self.pos_rewards

    @property
    def LBs(self):
        return self.lower_bounds

    @property
    def mchsStartTimes(self):
        return self.machine_start_times

    @property
    def opIDsOnMchs(self):
        return self.op_ids_on_machines
