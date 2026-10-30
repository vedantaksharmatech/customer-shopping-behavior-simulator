"""Create inactivity sessions and modeled state sequences."""
import pandas as pd
from .state_mapper import map_event


def build_sessions(events: pd.DataFrame, inactivity_minutes: int = 30) -> tuple[pd.DataFrame, dict[str, int]]:
    """Sessionize each visitor and terminate at purchase or inferred exit."""
    if inactivity_minutes <= 0:
        raise ValueError("inactivity_minutes must be positive")
    required = {"visitorid", "timestamp", "event"}
    if not required.issubset(events.columns):
        raise ValueError(f"Events must contain columns: {sorted(required)}")
    ordered = events.sort_values(["visitorid", "timestamp"], kind="stable").copy()
    gap = ordered.groupby("visitorid", sort=False)["timestamp"].diff()
    # Comparing seconds avoids NumPy's deprecated generic timedelta conversion.
    ordered["_new_session"] = gap.isna() | (gap.dt.total_seconds() > inactivity_minutes * 60)
    ordered["_session_number"] = ordered.groupby("visitorid", sort=False)["_new_session"].cumsum()
    rows: list[dict[str, object]] = []
    starts_with_purchase = 0
    for (visitor, number), group in ordered.groupby(["visitorid", "_session_number"], sort=False):
        states = [map_event(event) for event in group["event"]]
        if states and states[0] == "PURCHASED":
            starts_with_purchase += 1
        if "PURCHASED" in states:
            states = states[: states.index("PURCHASED") + 1]
            outcome = "PURCHASED"
        else:
            states.append("EXITED")
            outcome = "EXITED"
        rows.append({"session_id": f"{visitor}-{number}", "visitorid": visitor,
                     "start_time": group["timestamp"].iloc[0], "states": states,
                     "outcome": outcome, "event_count": len(group)})
    sessions = pd.DataFrame(rows, columns=["session_id", "visitorid", "start_time", "states", "outcome", "event_count"])
    stats = {"sessions_total": len(sessions), "sessions_starting_with_purchase": starts_with_purchase,
             "sessions_purchased": int((sessions["outcome"] == "PURCHASED").sum()) if len(sessions) else 0,
             "sessions_exited": int((sessions["outcome"] == "EXITED").sum()) if len(sessions) else 0}
    return sessions, stats
