"""Save project charts without opening interactive windows."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import STATE_ORDER


def save_charts(event_counts: pd.DataFrame, probabilities: pd.DataFrame, observed: pd.Series,
                simulated: pd.DataFrame, output: Path, transition_metrics: pd.DataFrame | None = None) -> None:
    """Generate event, transition, outcome, validation, and journey plots."""
    output.mkdir(parents=True, exist_ok=True)
    if not event_counts.empty:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(event_counts["event"], event_counts["count"])
        ax.set(title="Observed raw event distribution", xlabel="Raw event", ylabel="Event count"); fig.tight_layout()
        fig.savefig(output / "raw_event_distribution.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); image = ax.imshow(probabilities.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(4), STATE_ORDER, rotation=35, ha="right"); ax.set_yticks(range(4), STATE_ORDER)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{probabilities.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set(title="Estimated transition probabilities", xlabel="Next state", ylabel="Current state")
    fig.colorbar(image, ax=ax, label="Probability"); fig.tight_layout(); fig.savefig(output / "transition_heatmap.png", dpi=150); plt.close(fig)
    outcomes = ["PURCHASED", "EXITED", "STEP_LIMIT"]
    real = observed.value_counts(normalize=True).reindex(outcomes, fill_value=0)
    sim = simulated["outcome"].value_counts(normalize=True).reindex(outcomes, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(outcomes)); width = .36
    ax.bar(x - width/2, real.values, width, label="Observed holdout"); ax.bar(x + width/2, sim.values, width, label="Simulated")
    ax.set_xticks(x, real.index); ax.set(ylabel="Proportion", title="Observed versus simulated session outcomes"); ax.legend(); fig.tight_layout()
    fig.savefig(output / "outcome_comparison.png", dpi=150); plt.close(fig)
    if transition_metrics is not None and not transition_metrics.empty:
        fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(16)
        ax.plot(x, transition_metrics["observed_probability"].to_numpy(), "o", label="Observed test transitions")
        ax.plot(x, transition_metrics["simulated_probability"].to_numpy(), "x", label="Simulated transitions")
        ax.set_xticks(x, [f"{a}→{b}" for a in STATE_ORDER for b in STATE_ORDER], rotation=75, ha="right")
        ax.set(ylabel="Transition probability", title="Observed versus simulated transition probabilities"); ax.legend(); fig.tight_layout()
        fig.savefig(output / "transition_comparison.png", dpi=150); plt.close(fig)
    if not simulated.empty:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(simulated["journey_length"], bins=min(20, max(1, simulated["journey_length"].nunique())))
        ax.set(title="Simulated journey length distribution", xlabel="Transitions before stopping", ylabel="Journey count"); fig.tight_layout()
        fig.savefig(output / "simulated_journey_lengths.png", dpi=150); plt.close(fig)
