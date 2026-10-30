# Customer Shopping Behavior Simulator Using Markov Chains

A small, explainable Python project that estimates customer state transitions from Retailrocket event sessions and generates reproducible simulated journeys. It uses pandas, NumPy, and Matplotlib; no GPU, neural network, or Kaggle API credentials are needed.

## Problem and objectives

E-commerce clickstreams are event sequences. This project groups them into sessions, maps each event to a simple shopping state, estimates next-state probabilities from training sessions, simulates journeys, and compares simulated terminal outcomes with held-out sessions. Simulations are model outputs, not forecasts; validation determines how closely they resemble the supplied holdout data.

## Markov Chain idea

A first-order Markov Chain describes a state sequence where the next state is sampled from probabilities conditioned on the current state. Each matrix row gives the probabilities of moving from its row state to each column state. Rows with observed transitions sum to one. PURCHASED and EXITED are absorbing states so a journey stops at either outcome.

## Dataset

Use the [Retailrocket E-commerce Dataset on Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). Download and extract it manually. Put `events.csv` in `data/raw/events.csv`; see [data/README.md](data/README.md). Required columns are `timestamp`, `visitorid`, and `event`. `itemid` and `transactionid` are not required by this model. The loader expects Retailrocket Unix timestamps in milliseconds and validates the actual header. It excludes missing IDs/events, invalid timestamps, unsupported event types, and exact duplicate records (keeping the first copy); the data-quality table counts exclusions and raw event types. If more than half of records are excluded, modeling stops for review.

## Windows setup (PowerShell in VS Code)

Python 3.12.3 was available in the build environment. Python 3.11 is preferred by the original project brief; 3.12 is compatible with this project and its tested packages.

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip --version
python -m pip install -r requirements-dev.txt
```

The `requirements.txt` file pins pandas, NumPy, Matplotlib, and Streamlit. `requirements-dev.txt` adds pytest. `requirements-lock.txt` records the full resolved package set from the verified Python 3.12 environment; use `python -m pip install -r requirements-lock.txt` when you want that exact environment.

If activation is blocked by PowerShell policy, do not change a broad system policy. Invoke the local interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe run_project.py --help
.\.venv\Scripts\python.exe run_project.py
```

## Optional Streamlit dashboard

The dashboard uses the same loader, session builder, transition estimator, simulator, and validation functions as the CLI. Install its pinned dependency into this project's environment and launch it from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local address, normally `http://localhost:8501`, and may open it in your browser. Choose the CSV path and analysis settings in the sidebar, then select **Run analysis**. Stop the local server with `Ctrl+C` in the terminal.

Useful options include `--data`, `--inactivity-minutes`, `--journeys`, `--seed`, `--train-fraction`, `--sample-visitors`, `--max-sessions`, `--output-dir`, `--no-charts`, `--no-reports`, and `--synthetic-demo`. `--sample-visitors N` scans visitor IDs, samples identities, then retains every CSV row for those visitors, preserving complete histories and sessions. For a synthetic-only development run, use:

```powershell
.\.venv\Scripts\python.exe run_project.py --synthetic-demo
```

Synthetic data is explicitly labelled and must not be cited as a Retailrocket result. The default workflow requires real input and never silently falls back to synthetic data. `--max-sessions` is applied after sessionization, preserving complete sessions, but the first sessions are not a representative random sample. Prefer `--sample-visitors` when limiting memory before modeling.

## Pipeline and assumptions

1. Inspect columns, parse milliseconds since Unix epoch, count raw event types, and remove unusable/duplicate rows.
2. Sort each visitor's events and start a session after a configurable inactivity gap (30 minutes by default). This is an analytical convention, not ground truth.
3. Map `view`→`BROWSING`, `addtocart`→`CART`, and `transaction`→`PURCHASED`. Keep the sequence through its first purchase. If no purchase appears, append inferred `EXITED`. A transaction-first session begins at PURCHASED, with no fabricated prior transition.
4. Sort complete sessions by start time and split chronologically. Estimate transitions from training sessions only. Each transient row uses observed transition frequencies; a state without outgoing observations has a zero row and is not assigned invented probabilities.
5. Simulate from BROWSING with NumPy's seeded random generator. Each simulated record is labelled `simulated`; a step limit prevents runaway chains.
6. Compare terminal outcomes with the held-out test sessions and write CSVs and charts.

## Outputs

Generated below `outputs/` (ignored by Git):

- `tables/data_quality_summary.csv`, `raw_event_counts.csv`, `session_summary.csv`
- `tables/transition_counts.csv`, `transition_probabilities.csv`
- `tables/simulation_outcomes.csv`, `validation_metrics.csv`, `transition_validation_metrics.csv`, `sample_simulated_journeys.csv`
- `charts/raw_event_distribution.png`, `transition_heatmap.png`, `outcome_comparison.png`, `transition_comparison.png`, `simulated_journey_lengths.png`
- `reports/run_summary.md` with the source label, counts, fitted matrix, and holdout comparison

## Project structure

`src/shopping_simulator/` separates loading, cleaning, sessionization, state mapping, estimation, simulation, validation, visualization, reporting, and CLI code. `app.py` is the optional Streamlit dashboard and reuses those modules. `tests/` uses tiny fixtures and does not need the Kaggle file. `docs/` contains methodology, limitations, report outline, and viva questions.

## Interpretation and limits

The outcome comparison is a descriptive holdout check, not proof of predictive accuracy. An eventual-outcome calculation is available when the learned chain has an absorbing solution; it relies on the fitted matrix and its assumptions. The state abstraction omits products, prices, device, customer context, and timing except for session gaps. Visitor IDs may not correspond to people, session boundaries are inferred, first purchase ends the modeled journey, and the first-order assumption ignores longer history. Historical Retailrocket data may be selected or outdated. Read [docs/limitations.md](docs/limitations.md) before making claims.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Troubleshooting

- **Dataset not found:** copy Kaggle's `events.csv` to `data/raw/events.csv`, or pass `--data`.
- **Missing columns / many rows excluded:** inspect the source CSV header, timestamp encoding, and data-quality summary.
- **Unsupported transition state:** training has no outgoing examples for a state the simulator could visit. Use more complete sessions; the program intentionally does not invent a row.
- **PowerShell activation blocked:** use `.\.venv\Scripts\python.exe` directly.
- **Memory use:** begin with a valid full CSV and `--max-sessions` to cap modeled sessions. The loader currently reads the required three columns into memory; chunked cross-boundary sessionization is not implemented.
