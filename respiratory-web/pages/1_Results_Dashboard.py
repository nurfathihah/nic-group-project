"""📊 Results Dashboard — view the PSO vs GA comparison from precomputed artifacts.

This page does NO training and NO model loading. It only reads the small JSON/CSV
files the notebook exported, so it is fast and deploys reliably.
"""
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils import artifacts_exist, load_json, missing_artifacts_message

st.set_page_config(page_title="Results · BreatheAI", page_icon="📊", layout="wide")
st.title("📊 Results Dashboard")

if not artifacts_exist():
    missing_artifacts_message()
    st.stop()

summary = pd.read_csv("artifacts/results_summary.csv")
convergence = load_json("convergence.json")
cms = load_json("confusion_matrices.json")
hparams = load_json("best_hparams.json")

# ---------------------------------------------------------------- summary table
st.header("PSO vs GA — headline metrics")
st.caption(
    "Read **Macro F1** as the honest headline metric. Accuracy and Weighted F1 are "
    "inflated by the dominant Disease / COPD class and should not be read alone."
)
metric_cols = [c for c in summary.columns if summary[c].dtype != object]
st.dataframe(
    summary.style.format({c: "{:.4f}" for c in metric_cols})
                 .background_gradient(subset=["Macro F1"], cmap="Greens"),
    use_container_width=True, hide_index=True,
)

# ------------------------------------------------------------- best hyperparams
st.header("Best hyperparameters found")
for task, keys in [("Binary", ("pso_binary", "ga_binary")),
                   ("Multiclass", ("pso_multiclass", "ga_multiclass"))]:
    st.subheader(task)
    a, b = st.columns(2)
    with a:
        st.markdown("**PSO**")
        st.json(hparams[keys[0]])
    with b:
        st.markdown("**GA**")
        st.json(hparams[keys[1]])

# ------------------------------------------------------------ convergence curves
st.header("Optimizer convergence")
st.caption("Best validation macro-F1 at each iteration / generation.")


def convergence_fig(pso_hist, ga_hist, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=pso_hist, mode="lines+markers", name="PSO"))
    fig.add_trace(go.Scatter(y=ga_hist, mode="lines+markers", name="GA"))
    fig.update_layout(title=title, xaxis_title="Iteration / Generation",
                      yaxis_title="Best validation macro-F1", height=380,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


c1, c2 = st.columns(2)
c1.plotly_chart(convergence_fig(convergence["pso_binary"], convergence["ga_binary"],
                                "Binary"), use_container_width=True)
c2.plotly_chart(convergence_fig(convergence["pso_multiclass"], convergence["ga_multiclass"],
                                "Multiclass"), use_container_width=True)

# ------------------------------------------------------------- confusion matrices
st.header("Confusion matrices")


def cm_fig(matrix, labels, title):
    fig = px.imshow(matrix, x=labels, y=labels, text_auto=True,
                    color_continuous_scale="Blues", aspect="auto",
                    labels=dict(x="Predicted", y="True", color="Count"))
    fig.update_layout(title=title, height=430, margin=dict(l=10, r=10, t=40, b=10))
    return fig


bl, ml = cms["binary_labels"], cms["multiclass_labels"]
r1c1, r1c2 = st.columns(2)
r1c1.plotly_chart(cm_fig(cms["pso_binary"], bl, "PSO — Binary"), use_container_width=True)
r1c2.plotly_chart(cm_fig(cms["ga_binary"], bl, "GA — Binary"), use_container_width=True)
r2c1, r2c2 = st.columns(2)
r2c1.plotly_chart(cm_fig(cms["pso_multiclass"], ml, "PSO — Multiclass"), use_container_width=True)
r2c2.plotly_chart(cm_fig(cms["ga_multiclass"], ml, "GA — Multiclass"), use_container_width=True)

st.divider()
st.info(
    "**How to read the multiclass result:** the dataset is dominated by COPD, so a model "
    "can score high accuracy while barely recognising minority diagnoses. The confusion "
    "matrices and macro-F1 reveal this — which is exactly why they are reported here.",
    icon="🔎",
)
