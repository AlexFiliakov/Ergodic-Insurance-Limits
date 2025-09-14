"""Worker functions for parallel execution tests.

This module contains standalone worker functions that can be safely
imported in multiprocessing spawned processes without triggering
import chain issues with scipy/numpy.
"""

from typing import Any, Dict


def simulate_path_for_parallel_test(seed: int) -> Dict[str, Any]:
    """Simulate a single path for parallel executor test.

    Module-level function for pickle compatibility in multiprocessing.
    Simplified to avoid numpy import issues in spawned processes.
    """
    # Use basic Python to avoid numpy import issues in child processes
    import random

    random.seed(seed)

    n_years = 20
    values = [1000000.0]

    for _ in range(n_years):
        # Simulate returns without numpy
        return_val = random.gauss(0.05, 0.15)
        values.append(values[-1] * (1 + return_val))

    return {
        "terminal_value": values[-1],
        "max_value": max(values),
        "min_value": min(values),
        "seed": seed,
    }
