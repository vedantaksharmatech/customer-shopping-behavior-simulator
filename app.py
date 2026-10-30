"""Optional Streamlit dashboard using the project's shared analysis modules."""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shopping_simulator.config import DEFAULT_DATA, STATE_ORDER
from shopping_simulator.data_loader import load_events
from shopping_simulator.data_cleaning import summarize_event_counts
from shopping_simulator.session_builder import build_sessions
from shopping_simulator.transition_estimator import estimate_transitions
from shopping_simulator.simulator import simulate_journeys
from shopping_simulator.validation import split_sessions, validation_metrics, transition_validation_metrics


st.set_page_config(page_title="Shopping Journey Simulator", page_icon="🛒", layout="wide")
st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"] { background: #F3F7FB; }
      [data-testid="stSidebar"] { background: #E8F0F7; }
      [data-testid="stHeader"] { background: rgba(243, 247, 251, 0.92); }
      [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D7E2EC;
        border-radius: 0.65rem;
        padding: 0.75rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Customer Shopping Behavior Simulator")
st.caption("Estimate state transitions from event sessions, simulate journeys, and compare them with held-out sessions.")

with st.sidebar:
    st.header("Analysis settings")
    data_path = st.text_input("Events CSV path", value=str(DEFAULT_DATA))
    threshold = st.number_input("Session inactivity threshold (minutes)", min_value=1, max_value=1440, value=30)
    sample_visitors = st.number_input("Visitor sample size (0 = all visitors)", min_value=0, value=0, step=1000)
    train_fraction = st.slider("Training fraction", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
    journey_count = st.select_slider("Simulated journeys", options=[100, 500, 1000, 10000], value=1000)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    run = st.button("Run analysis", type="primary", use_container_width=True)

st.info("Add the Retailrocket `events.csv` file to `data/raw/`, or choose another CSV above. The dashboard never downloads data. Use the synthetic CLI demo for a small development run.")

if run:
    try:
        events, quality = load_events(data_path, sample_visitors=int(sample_visitors) or None, seed=int(seed))
        if quality["rows_read"] and len(events) / quality["rows_read"] < 0.5:
            raise ValueError("Fewer than half of loaded rows passed cleaning. Review the CSV schema and timestamp values before analysis.")
        sessions, session_stats = build_sessions(events, int(threshold))
        if len(sessions) < 2:
            raise ValueError(f"At least two sessions are needed for a holdout comparison; found {len(sessions)}.")
        training, testing = split_sessions(sessions, float(train_fraction))
        counts, probabilities = estimate_transitions(training)
        unsupported = probabilities.index[probabilities.sum(axis=1) == 0].tolist()
        if unsupported:
            raise ValueError(f"Training data has no outgoing examples for {unsupported}. Add more visitor/session data; the dashboard does not invent probabilities.")
        simulated = simulate_journeys(probabilities, n=int(journey_count), seed=int(seed))
        outcomes = validation_metrics(testing["outcome"], simulated["outcome"])
        transition_metrics = transition_validation_metrics(testing["states"].tolist(), simulated["states"].tolist())
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
    else:
        st.subheader("Data and session summary")
        left, middle, right = st.columns(3)
        left.metric("Rows retained", f"{quality['rows_retained']:,}")
        middle.metric("Sessions", f"{len(sessions):,}")
        right.metric("Training / holdout", f"{len(training):,} / {len(testing):,}")
        st.caption(f"Data source: {data_path} · Events read: {quality['rows_read']:,} · Excluded: {quality['rows_excluded']:,} · Visitors sampled: {quality['sampled_visitor_count']}")
        with st.expander("Data quality details"):
            st.dataframe(pd.DataFrame([quality]), use_container_width=True, hide_index=True)
            st.dataframe(summarize_event_counts(events), use_container_width=True, hide_index=True)

        st.subheader("Estimated transition matrix")
        st.caption("Learned from training sessions; PURCHASED and EXITED are absorbing. A zero transition count means no such transition was observed.")
        st.dataframe(probabilities.style.format("{:.3f}"), use_container_width=True)
        st.dataframe(counts, use_container_width=True)

        st.subheader("Observed versus simulated outcomes")
        st.bar_chart(outcomes.set_index("outcome")[["observed_proportion", "simulated_proportion"]])
        st.dataframe(outcomes, use_container_width=True, hide_index=True)

        st.subheader("Observed versus simulated transitions")
        transition_metrics["transition"] = transition_metrics["from_state"] + " → " + transition_metrics["to_state"]
        st.bar_chart(transition_metrics.set_index("transition")[["observed_probability", "simulated_probability"]])
        st.caption("Transition frequencies are compared across held-out observed sessions and simulated journeys. Terminal self-loops are excluded because journeys stop at terminal states.")

        st.subheader("Simulated journey lengths")
        st.bar_chart(simulated["journey_length"].value_counts().sort_index().rename("journeys"))
        st.subheader("Example simulated journeys")
        sample = simulated.head(10).copy()
        sample["states"] = sample["states"].map(lambda sequence: " → ".join(sequence))
        st.dataframe(sample, use_container_width=True, hide_index=True)
        st.warning("Simulated results describe paths generated by this fitted model. They are not forecasts, and no real-data findings can be reported until the real CSV is supplied.")
else:
    st.write("Choose settings in the sidebar and select **Run analysis** after placing the CSV. The synthetic development example remains available from the terminal:")
    st.code(".\\.venv\\Scripts\\python.exe run_project.py --synthetic-demo", language="powershell")
