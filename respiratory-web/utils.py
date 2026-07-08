"""Shared helpers for the BreatheAI app: artifact loading, feature extraction, prediction.

The feature-extraction here is IDENTICAL to the training notebook — that is essential,
otherwise the models see differently-shaped inputs and predictions become meaningless.
"""
import os
import json
import numpy as np
import librosa
import streamlit as st

# artifacts/ lives next to this file (repo root)
ART = os.path.join(os.path.dirname(__file__), "artifacts")


def artifacts_exist():
    """True only if the folder and the key files the app needs are present."""
    needed = ["feature_config.json", "labels.json", "results_summary.csv"]
    return os.path.isdir(ART) and all(os.path.exists(os.path.join(ART, f)) for f in needed)


@st.cache_data
def load_json(name):
    with open(os.path.join(ART, name)) as f:
        return json.load(f)


@st.cache_data
def load_config():
    return load_json("feature_config.json")


@st.cache_resource
def load_model(filename):
    """Load a Keras model once and keep it in memory. TF is imported lazily so the
    Results page can run even if TensorFlow is slow/unavailable."""
    import tensorflow as tf
    return tf.keras.models.load_model(os.path.join(ART, filename))


def wav_to_melspec(source, cfg):
    """source can be a file path OR a file-like object (e.g. an uploaded file).
    Returns a normalised log-mel spectrogram of shape (N_MELS, IMG_W)."""
    y, _ = librosa.load(source, sr=cfg["SR"], duration=cfg["DURATION"])
    target = int(cfg["SR"] * cfg["DURATION"])
    y = np.pad(y, (0, target - len(y))) if len(y) < target else y[:target]
    m = librosa.feature.melspectrogram(
        y=y, sr=cfg["SR"], n_mels=cfg["N_MELS"], hop_length=cfg["HOP"]
    )
    m = librosa.power_to_db(m, ref=np.max)
    if m.shape[1] < cfg["IMG_W"]:
        m = np.pad(m, ((0, 0), (0, cfg["IMG_W"] - m.shape[1])))
    else:
        m = m[:, : cfg["IMG_W"]]
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    return m.astype("float32")


def predict(model, mel, labels, is_binary):
    """Return (predicted_label, {label: probability})."""
    x = mel[np.newaxis, ..., np.newaxis]
    p = model.predict(x, verbose=0)[0]
    if is_binary:
        p_disease = float(p[0])                      # sigmoid output = P(Disease)
        probs = {labels[0]: 1.0 - p_disease, labels[1]: p_disease}
    else:
        probs = {labels[i]: float(p[i]) for i in range(len(labels))}
    label = max(probs, key=probs.get)
    return label, probs


def model_spec(task, optimizer, labels_json):
    """Map a UI choice to (model_filename, label_list, is_binary)."""
    opt = optimizer.lower()
    if task == "Binary (Healthy vs Disease)":
        return f"model_{opt}_binary.keras", labels_json["binary"], True
    return f"model_{opt}_multiclass.keras", labels_json["multiclass"], False


def missing_artifacts_message():
    st.warning(
        "**No `artifacts/` folder found.**\n\n"
        "This app displays and serves models produced by the training notebook. "
        "Run the notebook in Colab first (`Run all`), download the `artifacts/` folder "
        "it creates, and place it in the repository root next to `Home.py`.",
        icon="⚠️",
    )
    st.code(
        "respiratory-web/\n"
        "├── Home.py\n"
        "├── utils.py\n"
        "├── requirements.txt\n"
        "├── pages/\n"
        "└── artifacts/        <-- put the notebook's output here\n"
        "    ├── results_summary.csv\n"
        "    ├── convergence.json\n"
        "    ├── confusion_matrices.json\n"
        "    ├── best_hparams.json\n"
        "    ├── feature_config.json\n"
        "    ├── labels.json\n"
        "    └── model_*.keras",
        language="text",
    )
