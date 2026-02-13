import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


type AdjacencyMatrix = NDArray[np.single]
"""(num_ops, num_ops) constraint graph adjacency matrix (float32).

Example (2 jobs x 2 machines, 4 ops total)::

    # Initial state: self-loops + job-order arcs
    array([[1, 0, 0, 0],   # op0 (job0, step0)
           [1, 1, 0, 0],   # op1 -> op0 (same job predecessor)
           [0, 0, 1, 0],   # op2 (job1, step0)
           [0, 0, 1, 1]])  # op3 -> op2

    # After scheduling, disjunctive (machine-order) arcs are added.
"""


type Features = NDArray[np.single]
"""(num_ops, 2) per-operation features: [normalized_LB, finished_mark].

Example (4 ops)::

    array([[0.32, 0.],   # op0: LB=0.32, not yet scheduled
           [0.65, 1.],   # op1: LB=0.65, already scheduled
           [0.18, 0.],   # op2: LB=0.18, not yet scheduled
           [0.50, 0.]])  # op3: LB=0.50, not yet scheduled
"""


type Omega = NDArray[np.int64]
"""(num_jobs,) next schedulable operation ID per job.

Example (3 jobs x 2 machines)::

    array([0, 2, 4])  # initial: first op of each job
    array([1, 2, 5])  # after scheduling op0 and op4
"""


type Mask = NDArray[np.bool_]
"""(num_jobs,) True if job is fully scheduled.

Example (3 jobs)::

    array([False, False, False])  # initial: no job finished
    array([True,  False, False])  # job 0 fully scheduled
"""


type ProcessingTimes = NDArray[np.single]
"""(num_jobs, num_machines) processing time of each operation.

Example (3 jobs x 3 machines)::

    array([[83, 65,  3],    # job 0: durations for step 0, 1, 2
           [69, 42, 64],    # job 1
           [27, 27, 18]])   # job 2
"""


type MachineAssignments = NDArray[np.intp]
"""(num_jobs, num_machines) machine ID (1-indexed) for each operation.

Example (3 jobs x 3 machines)::

    array([[3, 2, 1],   # job 0: step0 on machine 3, step1 on machine 2, ...
           [1, 2, 3],   # job 1
           [2, 1, 3]])  # job 2
"""


class StepResult(BaseModel):
    """Result of a single environment step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    adjacency_matrix: AdjacencyMatrix = Field(description="(num_ops, num_ops) constraint graph adjacency matrix after this step.")
    features: Features = Field(description="(num_ops, 2) per-operation features: [normalized_LB, finished_mark].")
    reward: float = Field(description="Negative change in makespan lower bound (minimizing makespan).")
    done: bool = Field(description="True if all operations have been scheduled.")
    omega: Omega = Field(description="(num_jobs,) next schedulable operation ID per job.")
    mask: Mask = Field(description="(num_jobs,) True if job is fully scheduled.")


class ResetResult(BaseModel):
    """Result of environment reset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    adjacency_matrix: AdjacencyMatrix = Field(description="(num_ops, num_ops) initial constraint graph adjacency matrix.")
    features: Features = Field(description="(num_ops, 2) initial per-operation features (all zeros).")
    omega: Omega = Field(description="(num_jobs,) first operation ID per job.")
    mask: Mask = Field(description="(num_jobs,) initial mask (all False).")


class JSSPInstance(BaseModel):
    """A JSSP problem instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    times: ProcessingTimes = Field(description="(num_jobs, num_machines) processing time of each operation.")
    machines: MachineAssignments = Field(description="(num_jobs, num_machines) machine ID (1-indexed) for each operation.")
