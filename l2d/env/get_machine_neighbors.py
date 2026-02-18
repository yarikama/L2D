"""Adjacency matrix update helpers for the JSSP constraint graph."""

import numpy as np


def get_machine_neighbors(
    action: int,
    op_ids_on_machines: np.ndarray
) -> tuple[int, int]:
    """
    Find the predecessor and successor of an operation on its assigned machine.

    In the machine's processing order, find which operations come immediately
    before and after the given action.

    Args:
        action: The operation ID just scheduled.
        op_ids_on_machines: (n_m, n_j) array tracking which operations are
            assigned to each machine, in order. Negative values = empty slots.

    Returns:
        predecessor: Operation scheduled before `action` on the same machine,
            or `action` itself if it's the first.
        successor: Operation scheduled after `action` on the same machine,
            or `action` itself if it's the last (or no successor yet).
    """
    row, col = np.where(op_ids_on_machines == action)
    r, c = row.item(), col.item()

    # Predecessor: previous slot on same machine (or self if first)
    if c > 0:  # noqa: SIM108
        predecessor = op_ids_on_machines[r, c - 1].item()
    else:
        predecessor = action

    # Successor: next slot on same machine (or self if last/empty)
    if c + 1 < op_ids_on_machines.shape[1]:
        successor_candidate = op_ids_on_machines[r, c + 1].item()
        successor = action if successor_candidate < 0 else successor_candidate
    else:
        successor = action

    return predecessor, successor
