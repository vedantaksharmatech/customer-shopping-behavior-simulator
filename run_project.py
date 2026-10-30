"""Run the customer shopping behavior simulator from the project root."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from shopping_simulator.main import main

if __name__ == "__main__":
    raise SystemExit(main())
