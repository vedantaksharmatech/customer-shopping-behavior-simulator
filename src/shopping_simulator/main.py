"""Command-line workflow for real data or explicitly synthetic development data."""
import argparse
from pathlib import Path
import time
import pandas as pd
from .config import DEFAULT_DATA, DEFAULT_OUTPUTS, DEFAULT_SEED
from .data_cleaning import summarize_event_counts
from .data_loader import load_events
from .session_builder import build_sessions
from .transition_estimator import estimate_transitions
from .simulator import simulate_journeys
from .validation import split_sessions, validation_metrics, transition_validation_metrics, eventual_outcomes
from .visualizations import save_charts
from .reporting import write_table, write_run_report


def _synthetic_events() -> pd.DataFrame:
    """Tiny deterministic fixture for code development only, never real findings."""
    records = []
    examples = [["view", "view", "addtocart", "transaction"], ["view", "addtocart"],
                ["view", "transaction"], ["addtocart", "transaction"], ["view", "view"]]
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for visitor, sequence in enumerate(examples, start=1):
        for offset, event in enumerate(sequence):
            records.append({"visitorid": visitor, "timestamp": base + pd.Timedelta(minutes=offset), "event": event})
    return pd.DataFrame(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate and simulate customer shopping journeys with a Markov chain.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Input CSV (default: data/raw/events.csv)")
    parser.add_argument("--inactivity-minutes", type=int, default=30, help="Session gap threshold in minutes")
    parser.add_argument("--journeys", type=int, default=1000, help="Number of simulated journeys")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Chronological training fraction, between 0 and 1")
    parser.add_argument("--max-sessions", type=int, default=None, help="Development limit applied after complete sessions are built")
    parser.add_argument("--sample-visitors", type=int, default=None, help="Randomly sample visitor IDs and retain all their CSV events (memory-friendly, complete visitor histories)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS, help="Output root directory")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--no-reports", action="store_true", help="Skip the generated Markdown run summary")
    parser.add_argument("--synthetic-demo", action="store_true", help="Run a clearly labelled tiny synthetic development example")
    return parser


def run_workflow(events: pd.DataFrame, quality: dict[str, object], args: argparse.Namespace, source_label: str) -> None:
    started = time.perf_counter()
    sessions, session_stats = build_sessions(events, args.inactivity_minutes)
    if args.max_sessions is not None:
        if args.max_sessions < 2:
            raise ValueError("--max-sessions must be at least 2 to reserve a held-out session")
        sessions = sessions.iloc[:args.max_sessions].copy()
        print(f"Development limit: using first {len(sessions)} complete sessions; results may be unrepresentative.")
    if len(sessions) < 2:
        raise ValueError(f"Need at least two complete sessions; found {len(sessions)}")
    train, test = split_sessions(sessions, args.train_fraction)
    train_counts, probabilities = estimate_transitions(train)
    # Unseen transient states have no empirical row. Simulation refuses unsupported rows rather than inventing probabilities.
    if (probabilities.sum(axis=1) == 0).any():
        missing = probabilities.index[probabilities.sum(axis=1) == 0].tolist()
        raise ValueError(f"Training data has no outgoing transitions for {missing}. Increase training data or inspect session coverage; no probabilities were fabricated.")
    simulated = simulate_journeys(probabilities, args.journeys, args.seed)
    metrics = validation_metrics(test["outcome"], simulated["outcome"])
    transition_metrics = transition_validation_metrics(test["states"].tolist(), simulated["states"].tolist())
    outputs = args.output_dir
    tables = outputs / "tables"
    quality["source_label"] = source_label
    quality["sessions_total_before_limit"] = session_stats["sessions_total"]
    quality["sessions_used"] = len(sessions)
    quality["training_sessions"] = len(train)
    quality["test_sessions"] = len(test)
    write_table(pd.DataFrame([quality]), tables / "data_quality_summary.csv")
    write_table(summarize_event_counts(events), tables / "raw_event_counts.csv")
    write_table(pd.DataFrame([{**session_stats, "sessions_used": len(sessions), "training_sessions": len(train), "test_sessions": len(test)}]), tables / "session_summary.csv")
    write_table(train_counts, tables / "transition_counts.csv", index=True)
    write_table(probabilities, tables / "transition_probabilities.csv", index=True)
    write_table(metrics, tables / "validation_metrics.csv")
    write_table(transition_metrics, tables / "transition_validation_metrics.csv")
    write_table(simulated.groupby("outcome").size().rename("count").reset_index().assign(label="simulated"), tables / "simulation_outcomes.csv")
    journeys = simulated.head(20).copy()
    journeys["states"] = journeys["states"].map(lambda value: " -> ".join(value))
    write_table(journeys, tables / "sample_simulated_journeys.csv")
    if not args.no_charts:
        save_charts(summarize_event_counts(events), probabilities, test["outcome"], simulated, outputs / "charts", transition_metrics)
    if not args.no_reports:
        write_run_report(outputs / "reports" / "run_summary.md", source_label, quality, len(sessions),
                         len(train), len(test), args.journeys, args.seed, args.inactivity_minutes,
                         metrics, probabilities)
    print(f"Source: {source_label}. Retained {quality.get('rows_retained', len(events))} events; {len(sessions)} sessions; train={len(train)}, test={len(test)}.")
    print("Training transition probabilities:")
    print(probabilities.to_string(float_format=lambda value: f"{value:.3f}"))
    print("Observed test outcomes compared with simulated outcomes:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    try:
        print(f"Analytical eventual outcomes from BROWSING: {eventual_outcomes(probabilities)}")
    except ValueError as exc:
        print(f"Analytical eventual outcome unavailable: {exc}")
    print(f"Outputs saved in: {outputs.resolve()}")
    print(f"Elapsed: {time.perf_counter() - started:.2f} seconds")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.inactivity_minutes <= 0 or args.journeys <= 0:
            parser.error("--inactivity-minutes and --journeys must be positive")
        if args.max_sessions is not None and args.max_sessions <= 0:
            parser.error("--max-sessions must be positive")
        if args.sample_visitors is not None and args.sample_visitors <= 0:
            parser.error("--sample-visitors must be positive")
        if args.synthetic_demo:
            events = _synthetic_events()
            run_workflow(events, {"source": "in-memory development fixture", "rows_read": len(events),
                                  "rows_retained": len(events), "rows_excluded": 0}, args, "SYNTHETIC DEVELOPMENT DATA")
        else:
            events, quality = load_events(args.data, sample_visitors=args.sample_visitors, seed=args.seed)
            retention = len(events) / quality["rows_read"] if quality["rows_read"] else 0
            if retention < 0.5:
                raise ValueError(f"Only {retention:.1%} of rows were retained. Review the quality summary and source columns before modeling.")
            run_workflow(events, quality, args, "Retailrocket input file; results reflect supplied data")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 2
    return 0
