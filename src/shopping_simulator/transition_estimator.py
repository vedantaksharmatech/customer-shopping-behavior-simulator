"""Frequency-based Markov transition estimation."""
import numpy as np
import pandas as pd
from .config import STATE_ORDER


def estimate_transitions(sessions: pd.DataFrame, state_order: tuple[str, ...] = STATE_ORDER) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count observed transitions and estimate probabilities; terminals absorb."""
    counts = pd.DataFrame(0, index=state_order, columns=state_order, dtype=int)
    for sequence in sessions.get("states", []):
        for origin, destination in zip(sequence, sequence[1:]):
            counts.loc[origin, destination] += 1
    probabilities = pd.DataFrame(0.0, index=state_order, columns=state_order)
    for state in state_order:
        if state in {"PURCHASED", "EXITED"}:
            probabilities.loc[state, state] = 1.0
        else:
            total = counts.loc[state].sum()
            if total:
                probabilities.loc[state] = counts.loc[state] / total
    if counts.shape != (4, 4) or probabilities.shape != (4, 4):
        raise ValueError("Transition matrices must be 4 by 4")
    for state in ("PURCHASED", "EXITED"):
        counts.loc[state] = 0
    row_sums = probabilities.loc[[s for s in state_order if s not in {"PURCHASED", "EXITED"}]].sum(axis=1)
    if not np.isfinite(probabilities.to_numpy()).all() or (row_sums > 0).any() and not np.allclose(row_sums[row_sums > 0], 1):
        raise ValueError("Invalid transition probabilities")
    return counts, probabilities
