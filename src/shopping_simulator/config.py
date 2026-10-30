"""Shared project defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "raw" / "events.csv"
DEFAULT_OUTPUTS = ROOT / "outputs"
DEFAULT_THRESHOLD_MINUTES = 30
DEFAULT_SEED = 42
STATE_ORDER = ("BROWSING", "CART", "PURCHASED", "EXITED")
