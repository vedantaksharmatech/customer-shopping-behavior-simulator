import pandas as pd
from shopping_simulator.validation import split_sessions, validation_metrics, transition_validation_metrics, eventual_outcomes
from shopping_simulator.transition_estimator import estimate_transitions

def test_split_keeps_sessions_disjoint():
    sessions = pd.DataFrame({"session_id": ["a", "b", "c", "d"], "start_time": pd.date_range("2024-01-01", periods=4), "states": [["BROWSING", "EXITED"]] * 4, "outcome": ["EXITED"] * 4})
    train, test = split_sessions(sessions, .75)
    assert set(train.session_id).isdisjoint(test.session_id)
    assert len(train) == 3 and len(test) == 1

def test_validation_metric_and_eventual_outcomes_known_example():
    result = validation_metrics(pd.Series(["PURCHASED", "EXITED"]), pd.Series(["PURCHASED", "PURCHASED"]))
    assert result.set_index("outcome").loc["PURCHASED", "absolute_error"] == .5
    probabilities = estimate_transitions(pd.DataFrame({"states": [["BROWSING", "PURCHASED"], ["BROWSING", "EXITED"]]}))[1]
    assert eventual_outcomes(probabilities) == {"PURCHASED": .5, "EXITED": .5}

def test_transition_validation_compares_observed_with_simulated():
    result = transition_validation_metrics([["BROWSING", "PURCHASED"]], [["BROWSING", "EXITED"]])
    row = result[(result.from_state == "BROWSING") & (result.to_state == "PURCHASED")].iloc[0]
    assert row.observed_probability == 1 and row.simulated_probability == 0 and row.absolute_error == 1
