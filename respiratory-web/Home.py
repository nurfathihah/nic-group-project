"""BreatheAI — Respiratory Condition Detection (PSO vs GA tuned CNN).

Main entry point. Run with:  streamlit run Home.py
Streamlit auto-discovers the two files in pages/ and builds the sidebar nav.
"""
import streamlit as st
from utils import artifacts_exist, load_json, missing_artifacts_message

st.set_page_config(page_title="BreatheAI", page_icon="🫁", layout="wide")

st.title("🫁 BreatheAI")
st.subheader("Respiratory condition detection from breathing sounds — CNN tuned by PSO vs GA")

st.markdown(
    """
This app accompanies the study comparing **Particle Swarm Optimization (PSO)** and a
**Genetic Algorithm (GA)** for tuning a CNN that classifies respiratory recordings from
the ICBHI 2017 Respiratory Sound Database.

Use the sidebar to navigate:

- **📊 Results Dashboard** — view the PSO vs GA comparison: metrics, convergence curves,
  confusion matrices, and the best hyperparameters each optimizer found.
- **🔍 Try It Yourself** — upload your own breathing recording(s) and get a prediction
  from the trained models (single file or bulk).
"""
)

st.divider()

if not artifacts_exist():
    missing_artifacts_message()
else:
    labels = load_json("labels.json")
    c1, c2, c3 = st.columns(3)
    c1.metric("Task 1", "Binary", "Healthy vs Disease")
    c2.metric("Task 2", "Multiclass", f"{len(labels['multiclass'])} diagnoses")
    c3.metric("Optimizers compared", "PSO vs GA")
    st.success("Artifacts loaded. Head to **Results Dashboard** or **Try It Yourself** in the sidebar.", icon="✅")

st.divider()
st.caption(
    "Methodology: patient-grouped splits (no data leakage), macro-F1 as the imbalance-aware "
    "selection and reporting metric. Predictions are for research/education only — not a medical diagnosis."
)
