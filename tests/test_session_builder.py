import pandas as pd
from shopping_simulator.session_builder import build_sessions

def test_visitors_gap_boundary_and_terminal_policy():
    events = pd.DataFrame([
        (1, "2024-01-01T00:00:00Z", "view"), (1, "2024-01-01T00:10:00Z", "addtocart"),
        (1, "2024-01-01T01:00:00Z", "transaction"), (2, "2024-01-01T00:01:00Z", "addtocart"),
        (3, "2024-01-01T00:01:00Z", "view"), (3, "2024-01-01T00:02:00Z", "transaction"),
        (3, "2024-01-01T00:03:00Z", "view"), (4, "2024-01-01T00:01:00Z", "transaction")], columns=["visitorid", "timestamp", "event"])
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    sessions, stats = build_sessions(events, 30)
    assert len(sessions) == 5
    assert sessions.iloc[0]["states"] == ["BROWSING", "CART", "EXITED"]
    assert sessions.iloc[0]["outcome"] == "EXITED"
    purchase = sessions[sessions.visitorid == 3].iloc[0]
    assert purchase.states == ["BROWSING", "PURCHASED"]
    assert sessions[sessions.visitorid == 4].iloc[0].states == ["PURCHASED"]
    assert stats["sessions_starting_with_purchase"] == 2

def test_state_mapping_supports_three_events():
    from shopping_simulator.state_mapper import map_event
    assert [map_event(name) for name in ["view", "addtocart", "transaction"]] == ["BROWSING", "CART", "PURCHASED"]
