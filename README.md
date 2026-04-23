# SentinelEDU — Real-Time Student Performance Monitoring & Early Intervention

A capstone-grade, end-to-end intelligent system that monitors student academic
and behavioral data, predicts future performance, classifies risk, processes
streaming updates, triggers early warning alerts, and generates actionable
support recommendations for advisors.

---

## Quick Start (5 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize DVC and run the full pipeline (loads data, preprocesses, trains all models, evaluates)
dvc init --no-scm  # only if .dvc doesn't exist yet
dvc repro

# 3. Seed the SQLite database
python -m src.db.seed

# 4. Start the FastAPI backend (terminal 1)
uvicorn src.api.main:app --reload --port 8000

# 5. Start the Streamlit dashboard (terminal 2)
streamlit run dashboard/app.py
```

Then open:
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## What This Project Delivers

| Required Topic                                    | Where It Lives                                           |
|---------------------------------------------------|----------------------------------------------------------|
| CNN / Vanilla Deep Learning                       | `src/models/cnn.py` (1D CNN on sequential features)      |
| Streaming + Predictive Maintenance (adapted)      | `src/streaming/` + regression-based alerts               |
| Multivariate Linear Regression                    | `src/models/regression.py`                               |
| Non-Parametric Models                             | KNN + Decision Tree in `src/models/`                     |
| Cross-Validation                                  | `src/evaluation/cross_validation.py` (5-fold stratified) |
| Classification Performance Metrics                | `src/evaluation/metrics.py`                              |
| K-Nearest Neighbors                               | `src/models/regression.py` + `src/models/classification.py` |
| Logistic Regression                               | `src/models/classification.py`                           |
| Single Layer Perceptron                           | `src/models/deep.py`                                     |
| Multi Layer Perceptron                            | `src/models/deep.py`                                     |
| Gradient Descent                                  | Loss curves + lr comparison in training                  |
| Fine Tuning ANN                                   | `src/models/deep.py` (3-config tuning)                   |
| Data Streaming Visualization                      | Live alert timeline in Streamlit dashboard               |

---

## Project Structure

```
sentinel-edu/
├── config/                  # YAML configs (settings, alert thresholds, model registry)
├── data/                    # raw, processed, streaming data
├── artifacts/               # fitted preprocessor, scalers, encoders
├── models/                  # trained model artifacts by task
├── reports/                 # figures + model comparison reports
├── src/                     # application source
│   ├── data/                # loaders, validators, augmentors
│   ├── preprocessing/       # feature pipelines
│   ├── models/              # model implementations
│   ├── training/            # training scripts (orchestrated by DVC)
│   ├── evaluation/          # metrics, CV, plotting
│   ├── inference/           # inference service + model registry loader
│   ├── streaming/           # producer, consumer, feature store
│   ├── alerts/              # alert engine + rules
│   ├── recommendations/     # recommendation engine
│   ├── api/                 # FastAPI backend
│   ├── db/                  # SQLAlchemy models + seed
│   └── utils/               # logging + helpers
├── dashboard/               # Streamlit app
│   ├── app.py               # entry point
│   ├── pages/               # multi-page dashboard
│   ├── components/          # reusable UI components
│   └── theme/               # custom CSS
├── notebooks/               # EDA + experimentation
├── dvc.yaml                 # DVC pipeline definition
├── requirements.txt
└── README.md
```

---

## DVC Pipeline

The DVC pipeline has five stages defined in `dvc.yaml`:

1. **load_data** — downloads/validates UCI Student Performance data
2. **preprocess** — fits preprocessor, generates features, creates splits
3. **train_classical** — Linear Regression, Ridge, KNN, Logistic, Decision Tree, K-Means
4. **train_deep** — SLP, MLP, Fine-Tuned ANN, 1D CNN
5. **evaluate** — generates all metrics, plots, and the model comparison report

Run the whole thing with `dvc repro`. Outputs are tracked under DVC.

---

## Architecture Summary

- **Data layer** unifies historical UCI records with simulated LMS streams.
- **Preprocessing layer** with persisted fit-once pipelines guarantees training-serving consistency.
- **Model layer** contains regression, classification, neural, and 1D convolutional models.
- **Inference orchestration layer** is a FastAPI service loading all models at startup.
- **Streaming layer** emits events through a background producer thread; a consumer updates rolling feature windows and triggers inference.
- **Alert engine** fires on configurable YAML rules (trend-based and threshold-based).
- **Recommendation engine** converts alerts + segments into ranked advisor actions.
- **Dashboard** is a Streamlit multi-page app that polls the API.

---

## Demo Flow

1. Start backend + dashboard.
2. Show Home page — KPI cards, risk distribution, empty alert timeline.
3. Click "Start Stream" on the Home page.
4. Watch alerts populate live as the streaming producer emits events.
5. Click into a High Risk student.
6. Walk through the profile: predictions, trends, alerts, recommendations.
7. Show Model Performance page for evaluation depth.

---

## Honest Limitations

- UCI dataset is small (~1000 rows) and regional. Models are bounded by this.
- Streaming is simulated. Real production would use Kafka/Redis Streams.
- Recommendations are rule-based. A real system would train on intervention outcomes.
- No fairness audit. Documented as future work.
