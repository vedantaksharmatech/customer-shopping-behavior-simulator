"""Session-level holdout validation and absorbing-chain analysis."""
import numpy as np
import pandas as pd
from .config import STATE_ORDER


def split_sessions(sessions: pd.DataFrame, train_fraction: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split complete sessions, keeping sessions intact."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    ordered = sessions.sort_values("start_time", kind="stable").reset_index(drop=True)
    if len(ordered) < 2:
        raise ValueError("At least two sessions are required for a train/test split")
    boundary = min(max(int(len(ordered) * train_fraction), 1), len(ordered) - 1)
    return ordered.iloc[:boundary].copy(), ordered.iloc[boundary:].copy()


def outcome_summary(outcomes: pd.Series) -> pd.DataFrame:
    """Return counts and proportions for terminal outcome labels."""
    counts = outcomes.value_counts().reindex(["PURCHASED", "EXITED", "STEP_LIMIT"], fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame({"outcome": counts.index, "count": counts.values,
                         "proportion": counts.values / total if total else np.zeros(len(counts))})


def validation_metrics(observed: pd.Series, simulated: pd.Series) -> pd.DataFrame:
    """Compare terminal-outcome proportions with absolute errors."""
    real = outcome_summary(observed).set_index("outcome")["proportion"]
    sim = outcome_summary(simulated).set_index("outcome")["proportion"]
    result = pd.DataFrame({"observed_proportion": real, "simulated_proportion": sim})
    result["absolute_error"] = (result["observed_proportion"] - result["simulated_proportion"]).abs()
    result.index.name = "outcome"
    return result.reset_index()


def transition_validation_metrics(observed_sequences: list[list[str]], simulated_sequences: list[list[str]]) -> pd.DataFrame:
    """Compare observed holdout and simulated transition frequencies, excluding terminal self-loops."""
    rows = []
    for label, sequences in (("observed", observed_sequences), ("simulated", simulated_sequences)):
        counts = {(origin, destination): 0 for origin in STATE_ORDER for destination in STATE_ORDER}
        for sequence in sequences:
            for origin, destination in zip(sequence, sequence[1:]):
                counts[(origin, destination)] += 1
        totals = {state: sum(counts[(state, target)] for target in STATE_ORDER) for state in STATE_ORDER}
        for origin in STATE_ORDER:
            for destination in STATE_ORDER:
                # Absorbing self-transitions describe the model, not a sampled step in a terminated journey.
                probability = counts[(origin, destination)] / totals[origin] if totals[origin] else 0.0
                row = next((item for item in rows if item["from_state"] == origin and item["to_state"] == destination), None)
                if row is None:
                    row = {"from_state": origin, "to_state": destination}
                    rows.append(row)
                row[f"{label}_probability"] = probability
    result = pd.DataFrame(rows)
    result["absolute_error"] = (result["observed_probability"] - result["simulated_probability"]).abs()
    return result


def eventual_outcomes(probabilities: pd.DataFrame, initial_state: str = "BROWSING") -> dict[str, float]:
    """Solve eventual purchase/exit probabilities for absorbing terminal states."""
    transient = [s for s in STATE_ORDER if s not in {"PURCHASED", "EXITED"}]
    p = probabilities.reindex(index=STATE_ORDER, columns=STATE_ORDER).to_numpy(dtype=float)
    q = p[:2, :2]
    r = p[:2, 2:]
    try:
        result = np.linalg.solve(np.eye(2) - q, r)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Eventual outcomes are undefined: transient states may never reach a terminal state") from exc
    if initial_state in {"PURCHASED", "EXITED"}:
        return {"PURCHASED": float(initial_state == "PURCHASED"), "EXITED": float(initial_state == "EXITED")}
    if initial_state not in transient:
        raise ValueError(f"Unknown initial state: {initial_state}")
    i = transient.index(initial_state)
    return {"PURCHASED": float(result[i, 0]), "EXITED": float(result[i, 1])}
