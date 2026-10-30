"""Read and validate the raw event CSV without modifying it."""
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "visitorid", "event"}
KNOWN_EVENTS = {"view", "addtocart", "transaction"}


def load_events(path: str | Path, sample_visitors: int | None = None, seed: int = 42) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a CSV and return validated event rows plus a quality summary."""
    source = Path(path)
    # Some archive extractors create an events.csv directory containing the actual events.csv.
    # Accept that layout so the project's default data path still works without moving the file.
    if source.is_dir() and (source / "events.csv").is_file():
        source = source / "events.csv"
    if not source.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {source}\nDownload the Retailrocket dataset from "
            "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset and place "
            "events.csv in data/raw/, or pass --data with its path."
        )
    if sample_visitors is not None and sample_visitors <= 0:
        raise ValueError("sample_visitors must be positive")
    try:
        header = pd.read_csv(source, nrows=0).columns.tolist()
        missing = REQUIRED_COLUMNS - set(header)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}. Found: {header}")
        # Only columns used by this model are retained, reducing memory on the full file.
        # Keep optional item/transaction IDs through duplicate detection so two distinct products
        # clicked at the same time are not mistaken for duplicate events.
        columns = [column for column in ("timestamp", "visitorid", "event", "itemid", "transactionid") if column in header]
        total_dataset_rows = sum(len(chunk) for chunk in pd.read_csv(source, usecols=["visitorid"], chunksize=250_000)) if sample_visitors else None
        sampled_visitors = None
        if sample_visitors:
            # Sample identities, then load every event for those identities so sessions remain intact.
            visitor_ids: set[object] = set()
            for chunk in pd.read_csv(source, usecols=["visitorid"], chunksize=250_000):
                visitor_ids.update(chunk["visitorid"].dropna().unique().tolist())
            if not visitor_ids:
                raise ValueError("No non-missing visitor IDs are available to sample")
            ids = np.asarray(list(visitor_ids), dtype=object)
            sampled_visitors = set(np.random.default_rng(seed).choice(ids, size=min(sample_visitors, len(ids)), replace=False).tolist())
            pieces = []
            for chunk in pd.read_csv(source, usecols=columns, chunksize=250_000):
                selected = chunk[chunk["visitorid"].isin(sampled_visitors)]
                if not selected.empty:
                    pieces.append(selected)
            frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
        else:
            frame = pd.read_csv(source, usecols=columns)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset CSV is empty: {source}") from exc
    rows_read = len(frame)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", errors="coerce", utc=True)
    frame["event"] = frame["event"].astype("string").str.strip().str.lower()
    event_counts = frame["event"].value_counts(dropna=False).to_dict()
    missing_visitor = frame["visitorid"].isna()
    missing_event = frame["event"].isna() | frame["event"].eq("")
    invalid_time = frame["timestamp"].isna()
    unknown = ~frame["event"].isin(KNOWN_EVENTS) & ~missing_event
    duplicate = frame.duplicated(keep="first")
    rejected = missing_visitor | missing_event | invalid_time | unknown | duplicate
    clean = frame.loc[~rejected, ["timestamp", "visitorid", "event"]].copy()
    clean = clean.sort_values(["visitorid", "timestamp"], kind="stable").reset_index(drop=True)
    summary: dict[str, object] = {
        "source": str(source), "rows_read": rows_read, "rows_retained": len(clean),
        "total_dataset_rows": total_dataset_rows if total_dataset_rows is not None else rows_read,
        "dataset_file_size_bytes": source.stat().st_size,
        "sampled_visitor_count": len(sampled_visitors) if sampled_visitors is not None else "full dataset",
        "available_columns": ", ".join(header),
        "optional_columns_present": ", ".join(column for column in ("itemid", "transactionid") if column in header),
        "rows_excluded": int(rejected.sum()), "missing_visitor_id": int(missing_visitor.sum()),
        "missing_event": int(missing_event.sum()), "invalid_timestamp": int(invalid_time.sum()),
        "unrecognized_event": int(unknown.sum()), "duplicate_rows_removed": int(duplicate.sum()),
        "event_counts_json": str({str(k): int(v) for k, v in event_counts.items()}),
    }
    return clean, summary
