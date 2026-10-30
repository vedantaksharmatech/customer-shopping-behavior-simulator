import pandas as pd
from shopping_simulator.transition_estimator import estimate_transitions

def test_counts_probabilities_and_absorbing_rows():
    sessions = pd.DataFrame({"states": [["BROWSING", "BROWSING", "CART", "PURCHASED"], ["BROWSING", "EXITED"]]})
    counts, probabilities = estimate_transitions(sessions)
    assert counts.loc["BROWSING", "BROWSING"] == 1
    assert counts.loc["BROWSING", "EXITED"] == 1
    assert probabilities.loc["BROWSING"].sum() == 1
    assert probabilities.loc["PURCHASED"].tolist() == [0, 0, 1, 0]
    assert probabilities.loc["EXITED"].tolist() == [0, 0, 0, 1]
