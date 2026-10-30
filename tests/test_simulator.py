import pandas as pd
import pytest
from shopping_simulator.simulator import simulate_journeys
from shopping_simulator.transition_estimator import estimate_transitions

def model():
    return estimate_transitions(pd.DataFrame({"states": [["BROWSING", "PURCHASED"], ["BROWSING", "EXITED"], ["CART", "EXITED"]]}))[1]

def test_reproducibility_valid_outcomes_and_absorption():
    first = simulate_journeys(model(), 100, 9)
    assert first.equals(simulate_journeys(model(), 100, 9))
    assert set(first.outcome) <= {"PURCHASED", "EXITED"}

def test_invalid_matrix_fails_usefully():
    matrix = pd.DataFrame(0.0, index=["BROWSING", "CART", "PURCHASED", "EXITED"], columns=["BROWSING", "CART", "PURCHASED", "EXITED"])
    with pytest.raises(ValueError, match="sum to 1"):
        simulate_journeys(matrix)
