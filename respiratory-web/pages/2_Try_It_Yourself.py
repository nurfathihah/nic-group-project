"""🔍 Try It Yourself — upload breathing recording(s) and get predictions.

This page runs INFERENCE only (loads a trained model and calls predict). It never
retrains — that keeps it light enough to run on Streamlit Community Cloud.
"""
import io
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (artifacts_exist, load_config, load_json, load_model,
                   wav_to_melspec, predict, model_spec, missing_artifacts_message)

st.set_page_config(page_title="Try It · BreatheAI", page_icon="🔍", layout="wide")
st.title("🔍 Try It Yourself")

if not artifacts_exist():
    missing_artifacts_message()
    st.stop()

cfg = load_config()
labels_json = load_json("labels.json")

# ------------------------------------------------------------------- controls
c1, c2, c3 = st.columns(3)
task = c1.selectbox("Task", ["Binary (Healthy vs Disease)", "Multiclass (diagnosis)"])
optimizer = c2.radio("Model (optimizer)", ["PSO", "GA"], horizontal=True)
mode = c3.radio("Mode", ["Single file", "Bulk"], horizontal=True)

model_file, labels, is_binary = model_spec(task, optimizer, labels_json)

st.caption(
    f"Using **{optimizer}**-tuned **{'binary' if is_binary else 'multiclass'}** model. "
    "Upload `.wav` breathing recordings (same format as the ICBHI dataset)."
)

if not is_binary:
    st.warning(
        "Heads-up: the multiclass model is trained on a heavily COPD-dominated dataset, "
        "so it tends to predict COPD. Read the full probability bar, not just the top label.",
        icon="⚠️",
    )


def run_one(uploaded_file, model):
    """Returns (label, probs, mel) for one uploaded file."""
    data = uploaded_file.read()
    mel = wav_to_melspec(io.BytesIO(data), cfg)
    label, probs = predict(model, mel, labels, is_binary)
    return label, probs, mel


# ----------------------------------------------------------------- single mode
if mode == "Single file":
    up = st.file_uploader("Upload one .wav file", type=["wav"], accept_multiple_files=False)
    if up is not None:
        with st.spinner("Loading model and analysing…"):
            model = load_model(model_file)
            label, probs, mel = run_one(up, model)

        left, right = st.columns([1, 1])
        with left:
            top_conf = probs[label]
            st.metric("Prediction", label, f"{top_conf*100:.1f}% confidence")
            prob_df = (pd.DataFrame({"class": list(probs), "probability": list(probs.values())})
                       .sort_values("probability", ascending=True))
            fig = px.bar(prob_df, x="probability", y="class", orientation="h",
                         range_x=[0, 1], text_auto=".2f")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="", xaxis_title="Probability")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("**Input mel spectrogram** (what the CNN sees)")
            sfig = px.imshow(mel, origin="lower", aspect="auto",
                             color_continuous_scale="Magma")
            sfig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="Time", yaxis_title="Mel")
            st.plotly_chart(sfig, use_container_width=True)

# ------------------------------------------------------------------- bulk mode
else:
    ups = st.file_uploader("Upload multiple .wav files", type=["wav"],
                           accept_multiple_files=True)
    if ups:
        model = load_model(model_file)
        rows, prog = [], st.progress(0.0)
        for i, up in enumerate(ups):
            try:
                label, probs, _ = run_one(up, model)
                row = {"file": up.name, "prediction": label,
                       "confidence": round(probs[label], 4)}
                # include each class probability as its own column
                row.update({f"P({k})": round(v, 4) for k, v in probs.items()})
                rows.append(row)
            except Exception as e:
                rows.append({"file": up.name, "prediction": "ERROR", "confidence": str(e)})
            prog.progress((i + 1) / len(ups))

        results = pd.DataFrame(rows)
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download results as CSV",
                           results.to_csv(index=False).encode(),
                           file_name="predictions.csv", mime="text/csv")

st.divider()
st.caption("Research/education only — not a medical diagnosis.")
