"""Domain-specific type aliases for the L2D project.

These are semantic aliases over np.ndarray to make function signatures
self-documenting. They do NOT enforce shape at runtime — they are purely
for readability and IDE support.
"""

from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

# ── Environment state ───────────────────────────────────────────────
AdjMatrix: TypeAlias = NDArray[np.single]
"""(num_ops, num_ops) constraint graph adjacency matrix (float32)."""

Features: TypeAlias = NDArray[np.single]
"""(num_ops, 2) per-operation features: [normalized_LB, finished_mark]."""

Omega: TypeAlias = NDArray[np.int64]
"""(num_jobs,) next schedulable operation ID per job."""

Mask: TypeAlias = NDArray[np.bool_]
"""(num_jobs,) True if job is fully scheduled."""

# ── JSSP instance data ─────────────────────────────────────────────
ProcessingTimes: TypeAlias = NDArray[np.single]
"""(num_jobs, num_machines) processing time of each operation."""

MachineAssignments: TypeAlias = NDArray[np.intp]
"""(num_jobs, num_machines) machine ID (1-indexed) for each operation."""

# ── Structured returns ──────────────────────────────────────────────
class StepResult(NamedTuple):
    adj: AdjMatrix
    features: Features
    reward: float
    done: bool
    omega: Omega
    mask: Mask


class ResetResult(NamedTuple):
    adj: AdjMatrix
    features: Features
    omega: Omega
    mask: Mask


class JSSPInstance(NamedTuple):
    """A JSSP problem instance."""
    times: ProcessingTimes
    machines: MachineAssignments
