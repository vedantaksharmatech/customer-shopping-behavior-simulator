"""Reproducible journey generation from an estimated transition matrix."""
import numpy as np
import pandas as pd
from .config import STATE_ORDER


def simulate_journeys(probabilities: pd.DataFrame, n: int = 1000, seed: int = 42,
                      initial_state: str = "BROWSING", max_steps: int = 100) -> pd.DataFrame:
    """Generate independent journeys and record their terminal outcomes."""
    if n <= 0 or max_steps <= 0:
        raise ValueError("n and max_steps must be positive")
    if initial_state not in STATE_ORDER:
        raise ValueError(f"Unknown initial state: {initial_state}")
    matrix = probabilities.reindex(index=STATE_ORDER, columns=STATE_ORDER).to_numpy(dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Transition matrix must be a finite, non-negative 4 by 4 matrix")
    sums = matrix.sum(axis=1)
    if not np.allclose(sums, 1.0):
        raise ValueError("Every transition probability row must sum to 1; estimate a model with supported rows first")
    rng = np.random.default_rng(seed)
    rows = []
    for journey_id in range(n):
        sequence = [initial_state]
        current = initial_state
        while current not in {"PURCHASED", "EXITED"} and len(sequence) < max_steps:
            current = str(rng.choice(STATE_ORDER, p=matrix[STATE_ORDER.index(current)]))
            sequence.append(current)
        outcome = current if current in {"PURCHASED", "EXITED"} else "STEP_LIMIT"
        rows.append({"journey_id": journey_id + 1, "states": sequence, "outcome": outcome,
                     "journey_length": len(sequence) - 1, "label": "simulated"})
    return pd.DataFrame(rows)
