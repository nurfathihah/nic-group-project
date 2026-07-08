
# 🫁 BreatheAI — Respiratory Condition Detection (PSO vs GA)

A Streamlit web app for detecting respiratory conditions from breathing sounds, using a
CNN whose hyperparameters are tuned by two metaheuristics — **Particle Swarm Optimization
(PSO)** and a **Genetic Algorithm (GA)** — and comparing the two. Built on the
[ICBHI 2017 Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database).

The app has two pages:

- **📊 Results Dashboard** — view the PSO vs GA comparison: metrics, optimizer convergence
  curves, confusion matrices, and the best hyperparameters each optimizer found.
- **🔍 Try It Yourself** — upload your own `.wav` breathing recording(s) and get a prediction
  from the trained models, either one file at a time or in bulk (with a downloadable CSV).

## How it works

Training and serving are separated:

1. **Training (offline, in Google Colab with a GPU):** the notebook converts each recording
   into a log-mel spectrogram, runs the PSO and GA searches over the CNN hyperparameters,
   retrains the best model for each task, and saves everything into an `artifacts/` folder.
2. **Serving (this app):** the app loads those saved artifacts. It never retrains — the
   Results page reads the exported metrics, and the Try It page runs inference only. This
   keeps it light enough to run in a Codespace or on Streamlit Community Cloud.

### Methodology notes

- **No data leakage** — recordings are split by *patient* (`StratifiedGroupKFold`), so no
  patient appears in more than one split.
- **Imbalance-aware** — the dataset is dominated by COPD / "Disease", so PSO and GA optimize
  **macro-F1** (not accuracy), and macro-F1 is reported as the headline metric alongside
  accuracy and weighted-F1.

## Project structure

```
respiratory-web/
├── Home.py                      # landing page (run this)
├── utils.py                     # shared feature extraction + model loading
├── requirements.txt
├── pages/
│   ├── 1_Results_Dashboard.py   # Page 1 — results
│   └── 2_Try_It_Yourself.py     # Page 2 — upload & predict
└── artifacts/                   # produced by the training notebook (see below)
    ├── results_summary.csv
    ├── convergence.json
    ├── confusion_matrices.json
    ├── best_hparams.json
    ├── feature_config.json
    ├── labels.json
    └── model_*.keras
```

## Setup

### 1. Generate the artifacts (once)

Run the training notebook in Google Colab (GPU runtime → *Run all*). Its final cell creates
`artifacts.zip`. Download it and unzip it into the repo root so you have an `artifacts/`
folder next to `Home.py`.

### 2. Install the requirements

```bash
cd respiratory-web
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run Home.py
```

Open the forwarded URL (in Codespaces, use the **Ports** tab → port `8501`). Without the
`artifacts/` folder the app still runs, but shows a "no artifacts found" message until you
add it.

## Tech stack

Streamlit · TensorFlow/Keras · librosa · scikit-learn · Plotly

## Disclaimer

For research and educational purposes only. This app does **not** provide a medical
diagnosis.
