import pandas as pd
import pytest
from pathlib import Path
from shopping_simulator.data_loader import load_events

def test_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download the Retailrocket"):
        load_events(tmp_path / "missing.csv")

def test_events_csv_inside_extracted_events_folder(tmp_path):
    folder = tmp_path / "events.csv"
    folder.mkdir()
    pd.DataFrame({"timestamp": [1000, 2000], "visitorid": [1, 2], "event": ["view", "view"]}).to_csv(folder / "events.csv", index=False)
    events, quality = load_events(folder)
    assert len(events) == 2
    assert Path(quality["source"]).name == "events.csv"

def test_required_columns_and_timestamp_sorting(tmp_path):
    bad = tmp_path / "bad.csv"; pd.DataFrame({"event": ["view"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_events(bad)
    good = tmp_path / "events.csv"
    pd.DataFrame({"timestamp": [2000, 1000], "visitorid": [1, 1], "event": ["view", "addtocart"], "itemid": [7, 8]}).to_csv(good, index=False)
    events, quality = load_events(good)
    assert events["event"].tolist() == ["addtocart", "view"]
    assert quality["rows_retained"] == 2
    assert "itemid" in quality["optional_columns_present"]

def test_duplicate_detection_keeps_distinct_items(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame({"timestamp": [1000, 1000, 1000], "visitorid": [1, 1, 1], "event": ["view"] * 3,
                  "itemid": [10, 11, 10], "transactionid": [None, None, None]}).to_csv(path, index=False)
    events, quality = load_events(path)
    assert len(events) == 2
    assert quality["duplicate_rows_removed"] == 1

def test_visitor_sample_keeps_complete_visitor_rows(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame({"timestamp": [1000, 2000, 3000, 4000], "visitorid": [1, 1, 2, 2],
                  "event": ["view", "transaction", "view", "addtocart"]}).to_csv(path, index=False)
    selected, summary = load_events(path, sample_visitors=1, seed=3)
    assert selected["visitorid"].nunique() == 1
    assert len(selected) == 2
    assert summary["total_dataset_rows"] == 4
