"""Small data cleaning helpers; primary validation is in data_loader."""
import pandas as pd


def summarize_event_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Return observed counts for recognized raw event types."""
    return events["event"].value_counts().rename_axis("event").reset_index(name="count")
