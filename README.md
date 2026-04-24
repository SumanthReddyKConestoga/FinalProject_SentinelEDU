<div align="center">

# 🛡️ SentinelEDU

### Real-Time Student Performance Monitoring & Early Intervention Platform

*An end-to-end intelligent system that predicts academic risk before it becomes academic failure.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-3.53-945DD6?style=flat-square&logo=dvc&logoColor=white)](https://dvc.org/)
[![License](https://img.shields.io/badge/License-Academic-blue?style=flat-square)](#)

</div>

---

## 📖 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Technology Stack](#-technology-stack)
5. [Quick Start](#-quick-start)
6. [Detailed Setup Guide](#-detailed-setup-guide)
7. [Machine Learning Models](#-machine-learning-models)
8. [Real-Time Streaming System](#-real-time-streaming-system)
9. [Alert & Recommendation Engine](#-alert--recommendation-engine)
10. [AI Advisor (RAG System)](#-ai-advisor-rag-system)
11. [Dashboard Walkthrough](#-dashboard-walkthrough)
12. [API Reference](#-api-reference)
13. [Project Structure](#-project-structure)
14. [Configuration](#-configuration)
15. [Performance Metrics](#-performance-metrics)
16. [Troubleshooting](#-troubleshooting)
17. [Limitations & Future Work](#-limitations--future-work)
18. [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

**SentinelEDU** is a capstone-grade, production-style intelligent early-warning system designed for academic institutions. It continuously monitors student academic and behavioral data, predicts future performance using a suite of machine learning and deep learning models, processes live streaming updates, fires threshold-based alerts, and generates prioritized, actionable support recommendations for academic advisors — all surfaced through a clean multi-page web dashboard.

### The Problem

In a typical school, by the time a failing grade appears on a report card, it is often too late to help the student. Advisors manually track hundreds of students and rely on gut instinct to spot problems. Subtle warning signs — a three-week quiz-score decline, dropping LMS logins, sporadic attendance — go unnoticed until they compound into course failure.

### The Solution

SentinelEDU watches every student simultaneously, 24/7. It fuses classical ML, deep learning, temporal pattern detection, and retrieval-augmented generation into a single coherent platform that:

- **Predicts** each student's final grade with Ridge Regression (MAE ≈ 0.8 / 20)
- **Classifies** risk into Low / Medium / High / Critical using a 3-layer MLP
- **Detects** sustained decline patterns the MLP would miss using a 1D Convolutional Neural Network over 8-week behavioral sequences
- **Segments** students into 4 behavioral groups using K-Means clustering
- **Streams** live events through a producer-consumer thread architecture
- **Alerts** advisors the moment configurable YAML-defined rules are violated
- **Recommends** prioritized interventions with severity-weighted scoring and 14-day fatigue prevention
- **Advises** via an AI chatbot grounded in a curated intervention knowledge base (FAISS + Claude Haiku 4.5)

---

## ✨ Key Features

| Capability | What It Does |
|---|---|
| 🎓 **UCI Dataset Integration** | Auto-downloads 649 Portuguese student records (33 features) from the UCI ML Repository |
| 🔄 **DVC Pipeline** | Five reproducible stages: load → preprocess → train classical → train deep → evaluate |
| 🧠 **10+ ML Models** | Ridge, Linear, Logistic, KNN, Decision Tree, Naive Bayes, K-Means, SLP, MLP, Tuned ANNs (×3), 1D-CNN |
| 📊 **5-Fold Cross-Validation** | Every classical model validated with stratified k-fold CV |
| ⚡ **Real-Time Streaming** | Producer/consumer threads simulate LMS events at configurable rate |
| 🚨 **Configurable Alert Engine** | 6 production rules defined in YAML — threshold, slope, and compound condition types |
| 🎯 **Priority-Ranked Recommendations** | Severity-weighted actions with intervention fatigue prevention |
| 🤖 **RAG-Powered AI Advisor** | FAISS vector search over 15 expert documents + Claude-generated advice |
| 📱 **Multi-Page Dashboard** | Home, Roster, Student Profile, Alerts, Model Performance, AI Advisor |
| 🗄️ **Persistent Storage** | SQLite + SQLAlchemy ORM with 5 relational tables |
| 📈 **Interactive Visualizations** | Plotly charts, gauges, radar plots, confusion matrices, segment distributions |
| 🔌 **REST API** | 20+ FastAPI endpoints with auto-generated OpenAPI docs |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA & TRAINING LAYER                             │
│                                                                             │
│   UCI ML Repository  ──►  DVC Pipeline ──►  Parquet Splits ──►  Models      │
│   (ucimlrepo)             (5 stages)        (train/val/test)    (.pkl +     │
│                                                                  .keras)    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE & STREAMING LAYER                        │
│                                                                             │
│   ┌─────────────┐    ┌──────────────┐    ┌─────────────┐   ┌────────────┐   │
│   │  Producer   │───►│    Queue     │───►│  Consumer   │──►│  Feature   │   │
│   │  (Thread)   │    │ (thread-safe)│    │  (Thread)   │   │   Store    │   │
│   └─────────────┘    └──────────────┘    └──────┬──────┘   └────────────┘   │
│                                                  │                          │
│                                                  ▼                          │
│                              ┌────────────────────────────────┐             │
│                              │  InferenceService              │             │
│                              │   • Ridge (G3 prediction)      │             │
│                              │   • MLP (risk class)           │             │
│                              │   • 1D-CNN (temporal pattern)  │             │
│                              │   • K-Means (segment)          │             │
│                              └────────────────┬───────────────┘             │
│                                               │                             │
│                   ┌───────────────────────────┼──────────────────────┐      │
│                   ▼                           ▼                      ▼      │
│            ┌──────────────┐         ┌──────────────────┐      ┌──────────┐  │
│            │ Alert Engine │────────►│ Recommendation   │      │  SQLite  │  │
│            │ (YAML rules) │         │ Engine           │◄────►│    DB    │  │
│            └──────────────┘         └──────────────────┘      └──────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API & UI LAYER                                 │
│                                                                             │
│         ┌──────────────────┐         ┌─────────────────────────┐            │
│         │    FastAPI       │◄────────┤  Streamlit Dashboard    │            │
│         │  (port 8001)     │  HTTP/  │  (port 8501)            │            │
│         │                  │   JSON  │                         │            │
│         │  • /students     │         │   • Home (KPIs)         │            │
│         │  • /alerts       │         │   • Roster              │            │
│         │  • /streaming    │         │   • Student Profile     │            │
│         │  • /rag/query    │         │   • Alerts              │            │
│         │  • /models/*     │         │   • Model Performance   │            │
│         └────────┬─────────┘         │   • AI Advisor (RAG)    │            │
│                  │                   └─────────────────────────┘            │
│                  ▼                                                          │
│         ┌──────────────────┐         ┌─────────────────────────┐            │
│         │   RAG Engine     │────────►│   Anthropic Claude API  │            │
│         │  (FAISS + MiniLM)│  HTTPS  │   (claude-haiku-4.5)    │            │
│         └──────────────────┘         └─────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

<table>
<tr><th>Category</th><th>Technology</th><th>Version</th><th>Purpose</th></tr>
<tr><td rowspan="2"><b>Language & Orchestration</b></td><td>Python</td><td>3.11</td><td>All application code</td></tr>
<tr><td>DVC</td><td>3.53.2</td><td>ML pipeline orchestration + reproducibility</td></tr>

<tr><td rowspan="4"><b>Data</b></td><td>UCI ML Repository</td><td>—</td><td>Raw student performance dataset</td></tr>
<tr><td>ucimlrepo</td><td>0.0.7</td><td>Programmatic dataset download</td></tr>
<tr><td>pandas</td><td>2.2.2</td><td>DataFrame operations</td></tr>
<tr><td>Parquet (pyarrow)</td><td>16.1.0</td><td>Fast columnar storage</td></tr>

<tr><td rowspan="3"><b>Classical ML</b></td><td>scikit-learn</td><td>1.5.1</td><td>Regression, classification, clustering, preprocessing</td></tr>
<tr><td>numpy</td><td>1.26.4</td><td>Numeric computing</td></tr>
<tr><td>Joblib</td><td>1.4.2</td><td>Model serialization (.pkl)</td></tr>

<tr><td><b>Deep Learning</b></td><td>TensorFlow / Keras</td><td>2.17.0</td><td>SLP, MLP, Tuned ANN, 1D-CNN</td></tr>

<tr><td rowspan="4"><b>RAG / AI</b></td><td>sentence-transformers</td><td>≥3.0</td><td>Text → 384-dim embeddings (all-MiniLM-L6-v2)</td></tr>
<tr><td>FAISS (faiss-cpu)</td><td>≥1.7.4</td><td>Vector similarity search</td></tr>
<tr><td>Anthropic SDK</td><td>≥0.34</td><td>Claude API client</td></tr>
<tr><td>Claude Haiku 4.5</td><td>claude-haiku-4-5-20251001</td><td>Response generation</td></tr>

<tr><td rowspan="5"><b>Backend</b></td><td>FastAPI</td><td>0.112.0</td><td>REST API framework</td></tr>
<tr><td>Uvicorn</td><td>0.30.5</td><td>ASGI HTTP server</td></tr>
<tr><td>Pydantic</td><td>2.8.2</td><td>Request/response validation</td></tr>
<tr><td>SQLAlchemy</td><td>2.0.32</td><td>ORM</td></tr>
<tr><td>SQLite</td><td>built-in</td><td>Persistent storage</td></tr>

<tr><td rowspan="3"><b>Frontend</b></td><td>Streamlit</td><td>1.37.1</td><td>Multi-page dashboard</td></tr>
<tr><td>Plotly</td><td>5.22.0</td><td>Interactive charts</td></tr>
<tr><td>requests</td><td>2.32.3</td><td>Dashboard → API HTTP client</td></tr>

<tr><td rowspan="2"><b>Config</b></td><td>PyYAML</td><td>6.0.2</td><td>YAML config + alert rules</td></tr>
<tr><td>python-dotenv</td><td>1.0.1</td><td>Secret management (.env)</td></tr>
</table>

---

## 🚀 Quick Start

> **Prerequisites:** Python 3.11+, ~2 GB free disk space, ~4 GB RAM. First-time pipeline run takes 10–20 minutes (TensorFlow training).

### The 8 Commands That Get You Running

```bash
# 1. Clone the repository
git clone https://github.com/SumanthReddyKConestoga/FinalProject_SentinelEDU.git
cd FinalProject_SentinelEDU

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Initialize DVC (skip if the .dvc/ folder already exists)
dvc init --no-scm

# 5. Run the full ML pipeline — downloads data, trains ALL models
dvc repro

# 6. Seed the SQLite database
python -m src.db.seed

# 7. Start the FastAPI backend (Terminal 1)
uvicorn src.api.main:app --reload --port 8001

# 8. Start the Streamlit dashboard (Terminal 2)
streamlit run dashboard/app.py
```

Then open:
- 🎨 **Dashboard:** http://localhost:8501
- 📜 **API Docs (Swagger UI):** http://localhost:8001/docs

---

## 📋 Detailed Setup Guide

### Why These Steps Are Needed

When you first clone the repository, the following are **not** included (they are generated by the pipeline and excluded via `.gitignore`):

| What's Missing | Where It Gets Created |
|---|---|
| Raw dataset (`data/raw/`) | Stage 1 — `dvc repro` downloads from UCI |
| Processed Parquet splits (`data/processed/`) | Stage 2 — `dvc repro` |
| Streaming events (`data/streaming/events.jsonl`) | Stage 2 — `dvc repro` |
| Fitted preprocessor (`artifacts/preprocessor.pkl`) | Stage 2 — `dvc repro` |
| Trained models (`models/`) | Stages 3 & 4 — `dvc repro` |
| FAISS vector index (`artifacts/rag_index/`) | Built on first API startup |
| Evaluation reports (`reports/`) | Stage 5 — `dvc repro` |
| SQLite database (`sentinel.db`) | `python -m src.db.seed` |

### Optional: Enable the AI Advisor

The AI Advisor page works in **template mode** without an API key, but to unlock full Claude-generated responses:

1. Get an API key from [console.anthropic.com](https://console.anthropic.com)
2. Create a `.env` file in the project root:
   ```env
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
   ```
3. Restart the FastAPI server

### Expected Outputs After `dvc repro`

```
✓ data/raw/student_performance.csv           # 649 rows × 33 columns
✓ data/processed/static_train.parquet        # 80% training split
✓ data/processed/static_val.parquet          # 10% validation split
✓ data/processed/static_test.parquet         # 10% test split
✓ data/processed/weekly.parquet              # 5192 weekly records
✓ data/streaming/events.jsonl                # Streaming event source
✓ artifacts/preprocessor.pkl                 # Fitted ColumnTransformer
✓ models/regression/*.pkl                    # 4 regression models
✓ models/classification/*.pkl                # 4 classification models
✓ models/deep/*.keras                        # SLP, MLP, 3 Tuned ANNs
✓ models/cnn/cnn1d.keras                     # 1D Convolutional network
✓ models/clustering/kmeans.pkl               # 4-cluster K-Means
✓ reports/*.json                             # All evaluation metrics
✓ reports/figures/*.png                      # Confusion matrices, loss curves, residuals
✓ reports/model_comparison.md                # Narrative comparison report
✓ config/model_registry.json                 # Production model selections
```

---

## 🧠 Machine Learning Models

### Production Models (Live in Inference Service)

#### 1️⃣ Ridge Regression — Grade Prediction

Predicts the final grade (G3, 0–20) from 33 preprocessed features.

- **Input:** 33 features (demographics, study habits, G1, G2, behavioral aggregates)
- **Target:** G3 (continuous, 0–20)
- **Why Ridge over Linear?** G1 and G2 are highly correlated; L2 regularization (α=1.0) prevents coefficient instability
- **Why Ridge over Neural Network?** With only 649 samples, a neural network would overfit; Ridge is interpretable and trains in milliseconds
- **Performance:** RMSE ≈ 0.835, MAE ≈ 0.67, R² ≈ 0.925

#### 2️⃣ MLP Classifier — Risk Classification

A 3-hidden-layer neural network that classifies students into Low / Medium / High / Critical risk.

```
Input (33) → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dense(4, Softmax)
```

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Sparse Categorical Cross-Entropy
- **Training:** Up to 60 epochs with early stopping (patience=10)
- **Why MLP over Logistic Regression?** Learns non-linear decision boundaries; student risk is not linearly separable
- **Performance:** Accuracy ≈ 0.80, Macro-F1 ≈ 0.80

#### 3️⃣ 1D Convolutional Neural Network — Temporal Pattern Detection

Slides a detection window across each student's 8-week behavioral sequence.

```
Input (8 × 5) → Conv1D(32, k=3) → Conv1D(64, k=3) → GlobalMaxPool1D → Dense(32) → Dropout(0.4) → Dense(3, Softmax)
```

- **Why CNN matters:** Two students with identical 75% average attendance look identical to the MLP, but the CNN distinguishes *steady 75%* from *90% → 60% → 50% declining trend* — the latter is an urgent intervention signal
- **Used in critical alerts:** When CNN classifies a student as High Risk for **3+ consecutive weeks**, it fires the system's highest-severity `cnn_sustained_risk` alert
- **Performance:** Accuracy ≈ 0.77, Macro-F1 ≈ 0.78

#### 4️⃣ K-Means Clustering — Student Segmentation

Unsupervised segmentation of students into 4 behavioral groups:

| Segment | Profile |
|---|---|
| 🟢 **Consistent High Performers** | High grades, high attendance, low failures, high LMS engagement |
| 🔵 **Improving Students** | Started low, showing positive week-over-week trends |
| 🟠 **Disengaged Students** | Low LMS logins, missing submissions — at risk due to disengagement |
| 🔴 **High-Risk Students** | Multiple converging risk factors |

### Additional Models (For Academic Completeness & Comparison)

| Model | Type | Purpose |
|---|---|---|
| Linear Regression | Regression | Baseline |
| KNN Regressor | Non-parametric regression | Comparison |
| Decision Tree Regressor | Non-parametric regression | Interpretability baseline |
| Logistic Regression | Classification | Linear baseline |
| KNN Classifier | Non-parametric classification | Comparison |
| Decision Tree Classifier | Non-parametric classification | Rule-extractable baseline |
| Gaussian Naive Bayes | Probabilistic classifier | Fast baseline |
| Single Layer Perceptron (SLP) | Deep learning | Foundational neural model |
| Tuned ANN × 3 configs | Deep learning | Hyperparameter sensitivity study |

All models are evaluated with **5-fold stratified cross-validation** and compared in `reports/model_comparison.md`.

---

## ⚡ Real-Time Streaming System

### Architecture

```
events.jsonl ──► StreamProducer ──► queue.Queue ──► StreamConsumer
 (JSONL file)    (daemon thread)   (thread-safe)    (daemon thread)
                                                          │
                                                          ▼
                                    FeatureStore (in-memory rolling windows)
                                                          │
                                                          ▼
                                    InferenceService (Ridge + MLP + CNN)
                                                          │
                                                          ▼
                                    Alert Engine ──► Recommendation Engine
                                                          │
                                                          ▼
                                          SQLite (predictions, alerts, recs)
```

### How It Works

- **StreamProducer** — Reads `events.jsonl` line by line at a configurable rate (default: 2 events/sec) and pushes each event into a shared `queue.Queue`. Loops infinitely to simulate continuous activity.
- **StreamConsumer** — Pulls events, updates the **FeatureStore** (a per-student `deque` of rolling 8-week windows), runs inference, writes `WeeklyRecord` + `Prediction` rows, and invokes the Alert Engine.
- **FeatureStore** — Hydrated from the database at startup; kept in RAM for microsecond lookups. Computes derived features on demand: means, latest values, and 3-week slopes.

### Why Threading + Queue (Not Kafka)

| Concern | Decision |
|---|---|
| Industrial-scale streaming | Kafka / Redis Streams would be correct in production |
| Demo / capstone setting | Python `threading.Queue` is zero-config and sufficient for 2–10 events/sec |
| Swap cost | Replacing `queue.Queue` with a Kafka consumer is a ~20-line change — the architecture is intentionally decoupled |

---

## 🚨 Alert & Recommendation Engine

### The 6 Production Alert Rules (`config/thresholds.yaml`)

| Rule ID | Trigger | Severity | Cooldown |
|---|---|---|---|
| `attendance_drop` | `weekly_attendance_pct < 70%` | Medium | 24h |
| `quiz_decline` | `quiz_score_slope_3 < -1.0` | Medium | 24h |
| `late_submissions` | `weekly_late_count ≥ 2` | Low | 48h |
| `predicted_final_low` | `regression_prediction < 10` | High | 24h |
| `classifier_high_risk` | `classifier_label == "High" AND confidence > 0.75` | High | 24h |
| `cnn_sustained_risk` | `cnn_high_risk_consecutive ≥ 3` | **Critical** | 12h |

### Three Condition Types

- **`threshold`** — single feature vs. value (operators: `lt`, `lte`, `gt`, `gte`, `eq`, `neq`)
- **`slope`** — same operators, applied to computed slope features (e.g., `quiz_score_slope_3`)
- **`compound`** — multiple sub-conditions AND'd together

### Recommendation Engine Logic

When an alert fires, the engine:

1. **Looks up actions** for that rule in `thresholds.yaml`
2. **Applies severity multiplier:** Critical × 1.00, High × 0.85, Medium × 0.60, Low × 0.30
3. **Checks intervention fatigue:** skips any action already recommended to this student in the last 14 days
4. **Writes prioritized `Recommendation` rows** to the database
5. **Returns actions sorted by priority descending**

### Example Cascade

```
[CNN detects 3 consecutive High-Risk weeks for S0042]
         │
         ▼
[Alert: cnn_sustained_risk, severity=critical]
         │
         ▼
[Recommendations generated]:
   • advisor_meeting              priority = 0.95 × 1.00 = 0.95  (Critical)
   • counseling_referral          priority = 0.80 × 1.00 = 0.80
   • academic_probation_review    priority = 0.75 × 1.00 = 0.75
```

---

## 🤖 AI Advisor (RAG System)

A Retrieval-Augmented Generation pipeline that gives advisors grounded, personalized advice.

### Pipeline

```
Advisor question + Student ID
         │
         ▼
[1] Embed question with sentence-transformers (all-MiniLM-L6-v2 → 384-dim vector)
         │
         ▼
[2] FAISS IndexFlatL2 search → top 3 most relevant documents from 15-doc knowledge base
         │
         ▼
[3] Load student profile from SQLite (risk class, attendance, quiz scores, active alerts)
         │
         ▼
[4] Build prompt: system + retrieved docs + student data + question
         │
         ▼
[5] Call Anthropic Claude API (claude-haiku-4-5-20251001, max_tokens=1024)
         │
         ▼
[6] Return grounded answer + source citations
```

### The Knowledge Base (15 Curated Documents, 8 Categories)

- 📅 **Attendance** — Low attendance protocols, chronic absenteeism strategies
- 📉 **Academic Performance** — Declining grades, grade recovery plans, late submissions
- 💻 **Engagement** — LMS disengagement recovery
- 🔴 **High Risk** — Comprehensive support protocol, sustained critical risk escalation
- 🧠 **Mental Health** — Recognizing crisis signals in academic data
- 📚 **Tutoring** — Making referrals that actually stick
- 👥 **Advisor Practices** — Meeting frameworks, probation conversations
- 🔬 **Model Explanations** — What the CNN/MLP detect, how segmentation works

### Why This Architecture

- **FAISS (local) over Pinecone (cloud):** no external service, no API key, runs entirely in-process
- **sentence-transformers (local) over OpenAI embeddings:** no per-query cost, no network latency
- **Claude Haiku 4.5 over local LLMs:** a 7B local model needs 8+ GB VRAM; Haiku gives GPT-4-class reasoning at <2 sec latency without a GPU
- **Template fallback:** if `ANTHROPIC_API_KEY` is unset, the system still returns structured, document-cited advice

---

## 📱 Dashboard Walkthrough

### 🏠 Home — Mission Control

KPI cards (Total Students, Unacked Alerts, High Risk, Critical Risk), live alert feed with auto-refresh (5 sec), risk distribution donut, alert severity bar chart, segment distribution. Dark gradient hero with pulsing "LIVE" indicator.

### 👥 Roster

Searchable, filterable table of all students. Filters: ID search, risk class, segment, school. Color-coded risk badges. Click-through to student profile.

### 👤 Student Profile

Full deep-dive: risk-colored hero header, 4 metric cards (attendance, quiz score, late count, LMS logins), weekly trend chart (dual-axis: attendance + quiz score + late count), predicted G3 timeline with pass-threshold line, active alerts with acknowledge buttons, prioritized recommendations with "Act" button that logs AdvisorAction.

### 🚨 Alerts Center

Global alert timeline. Filters: severity, acknowledged/unacknowledged, time limit. Bulk acknowledge. Alert severity bar chart. Click-through to student.

### 📊 Model Performance

Five tabs — Regression / Classification / Deep Learning / CNN / Clustering. Each with leaderboard tables, gauge charts (Accuracy, Precision, Recall, F1), radar multi-model comparison, confusion matrices, and embedded PNG figures (residuals, loss curves, silhouette plots). Full `model_comparison.md` rendered at bottom.

### 🤖 AI Advisor

Two-column layout: left panel loads student context and offers 8 quick questions; right panel is an interactive chat with Claude. Responses include expandable "Sources used" citations to the knowledge base.

---

## 🔌 API Reference

Base URL: `http://localhost:8001`  
Interactive docs: `http://localhost:8001/docs`

### Students
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/students` | List students (filter by `risk_class`, paginate with `limit`/`offset`) |
| `GET` | `/students/{id}` | Single student row |
| `GET` | `/students/{id}/profile` | Full profile (student + predictions + weekly + alerts + recs) |
| `GET` | `/students/{id}/weekly` | Weekly records |
| `GET` | `/students/{id}/alerts` | Alert history |
| `GET` | `/students/{id}/recommendations` | Current recommendations |
| `POST` | `/students/{id}/advisor-action` | Log an AdvisorAction |

### Alerts
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/alerts/recent?limit=N` | Recent alerts across all students |
| `POST` | `/alerts/{id}/acknowledge` | Mark alert acknowledged |

### Streaming
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/streaming/start` | Start producer + consumer threads |
| `POST` | `/streaming/stop` | Stop both threads |
| `GET` | `/streaming/status` | Running state, queue depth, events processed |

### Models & Segments
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/models/metrics` | All evaluation JSONs |
| `GET` | `/models/registry` | Which models are in production |
| `GET` | `/models/comparison` | Markdown comparison report |
| `GET` | `/models/figures` | List of PNG figure filenames |
| `GET` | `/segments` | Count of students per segment |

### RAG
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/rag/query` | `{question, student_id?}` → grounded answer + sources |
| `GET` | `/rag/recommend/{student_id}` | AI-generated recommendations for a specific student |

### Health
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | `{"status": "ok"}` |

---

## 📁 Project Structure

```
FinalProject_SentinelEDU/
├── config/
│   ├── settings.yaml            # App-wide settings (paths, ports, training hyperparams)
│   ├── thresholds.yaml          # Alert rules + recommendation actions
│   └── model_registry.json      # Production model selections (auto-updated)
│
├── data/                        # [gitignored] Generated by DVC
│   ├── raw/                     # UCI CSV
│   ├── processed/               # Parquet splits
│   └── streaming/               # events.jsonl
│
├── artifacts/                   # [gitignored] Fitted preprocessor + FAISS index
│
├── models/                      # [gitignored] Trained models (.pkl + .keras)
│   ├── regression/
│   ├── classification/
│   ├── deep/
│   ├── cnn/
│   └── clustering/
│
├── reports/                     # [gitignored] Evaluation outputs
│   ├── figures/                 # Confusion matrices, loss curves, etc.
│   ├── *_metrics.json           # Per-task metric dumps
│   └── model_comparison.md      # Narrative summary
│
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app entry point
│   │   └── routes/              # students, alerts, recommendations, streaming, rag, ...
│   ├── data/
│   │   ├── loader.py            # UCI download + synthetic fallback
│   │   └── augmentor.py         # Weekly behavioral synthesis
│   ├── preprocessing/
│   │   ├── preprocessor.py      # fit-once ColumnTransformer
│   │   ├── feature_engineer.py  # risk_class derivation + splits
│   │   └── sequence_builder.py  # CNN (8×5) sequences
│   ├── models/
│   │   ├── regression.py        # Ridge, Linear, KNN, Decision Tree
│   │   ├── classification.py    # Logistic, KNN, Decision Tree, Naive Bayes
│   │   ├── deep.py              # SLP, MLP, Tuned ANN configs
│   │   ├── cnn.py               # 1D Convolutional network
│   │   └── clustering.py        # K-Means + segment names
│   ├── training/                # DVC stage scripts (load, preprocess, train_*, evaluate)
│   ├── evaluation/              # Metrics, cross-validation, plots
│   ├── inference/
│   │   ├── registry.py          # Loads all production models at startup
│   │   └── service.py           # Thread-safe InferenceService
│   ├── streaming/
│   │   ├── producer.py          # Event emitter thread
│   │   ├── consumer.py          # Event processor thread
│   │   └── feature_store.py     # In-memory rolling windows
│   ├── alerts/
│   │   ├── engine.py            # AlertEngine with cooldowns
│   │   └── rules.py             # Threshold / slope / compound evaluators
│   ├── recommendations/
│   │   └── engine.py            # Severity-weighted, fatigue-aware
│   ├── rag/
│   │   ├── knowledge_base.py    # 15 curated intervention documents
│   │   ├── embedder.py          # sentence-transformers + TF-IDF fallback
│   │   ├── retriever.py         # FAISS + numpy fallback
│   │   ├── generator.py         # Anthropic Claude + template fallback
│   │   └── rag_engine.py        # Unified entry point
│   ├── db/
│   │   ├── models.py            # SQLAlchemy ORM (5 tables)
│   │   ├── session.py           # FastAPI session dependency
│   │   └── seed.py              # Populate DB after training
│   ├── utils/                   # Logging + helpers
│   └── config.py                # YAML loaders (lru_cache singletons)
│
├── dashboard/
│   ├── app.py                   # Entry point + sidebar + page router
│   ├── api_client.py            # requests wrapper over FastAPI endpoints
│   ├── components/              # Cards, charts, tables
│   ├── theme/custom.css         # Design tokens + component styles
│   └── pages/                   # 6 Streamlit pages
│
├── dvc.yaml                     # 5-stage DVC pipeline definition
├── params.yaml                  # Hyperparameters tracked by DVC
├── requirements.txt             # Pinned dependencies
├── sentinel.db                  # [gitignored] SQLite database
└── .env                         # [gitignored] ANTHROPIC_API_KEY
```

---

## ⚙️ Configuration

### `config/settings.yaml` — App Settings

Controls data paths, split ratios, training hyperparameters (epochs, batch size, patience, learning rate), streaming rate, API port, and database URL.

### `config/thresholds.yaml` — Alert Rules & Recommendations

Human-editable: administrators can add new rules or adjust thresholds without touching Python.

### `.env` — Secrets

```env
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### `params.yaml` — DVC-Tracked Hyperparameters

When values here change, DVC automatically invalidates and re-runs affected stages.

---

## 📈 Performance Metrics

Actual metrics from `reports/` after running the pipeline on UCI data:

### Regression (Predicting G3)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| 🥇 **Ridge Regression** | **0.835** | **0.673** | **0.925** |
| Linear Regression | 0.841 | 0.679 | 0.924 |
| Decision Tree | 1.184 | 0.956 | 0.850 |
| KNN | 1.479 | 1.172 | 0.766 |

### Classification (Risk Class)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| 🥇 **MLP** | **0.800** | **0.827** | **0.783** | **0.801** |
| Decision Tree | 0.783 | 0.790 | 0.794 | 0.791 |
| SLP | 0.783 | 0.816 | 0.755 | 0.778 |
| Logistic Regression | 0.767 | 0.777 | 0.750 | 0.761 |
| KNN | 0.767 | 0.766 | 0.766 | 0.766 |
| Naive Bayes | 0.750 | 0.756 | 0.783 | 0.760 |
| 1D CNN (sequences) | 0.770 | 0.792 | 0.765 | 0.776 |

### Clustering

| Metric | Value |
|---|---|
| Silhouette Score | 0.055 |
| Clusters (K) | 4 |
| Cluster sizes | 54 / 85 / 86 / 54 |

---

## 🧰 Troubleshooting

<details>
<summary><b>TensorFlow install fails</b></summary>

Try the CPU-only variant:
```bash
pip install tensorflow-cpu==2.17.0
```
</details>

<details>
<summary><b>DVC stage fails mid-pipeline</b></summary>

Run stages individually to see the exact error:
```bash
python -m src.training.stage_load_data
python -m src.training.stage_preprocess
python -m src.training.stage_train_classical
python -m src.training.stage_train_deep
python -m src.training.stage_evaluate
```
</details>

<details>
<summary><b>UCI download fails (offline / firewall)</b></summary>

The `DataLoader` has a built-in synthetic fallback that generates a UCI-schema-compatible dataset of 500 rows. It kicks in automatically if `ucimlrepo.fetch_ucirepo()` raises.
</details>

<details>
<summary><b>Streaming produces no alerts</b></summary>

Lower the thresholds in `config/thresholds.yaml` temporarily — e.g., set `attendance_drop` to `lt 85` instead of `lt 70` to force triggers during demo runs.
</details>

<details>
<summary><b>AI Advisor returns template responses</b></summary>

Make sure `.env` exists in the project root with `ANTHROPIC_API_KEY=...` and **restart the FastAPI server** (dotenv only loads at process startup).
</details>

<details>
<summary><b>Streamlit "connection refused" errors</b></summary>

The dashboard's `api_client.py` is pointed at `http://localhost:8001`. Confirm:
1. FastAPI is actually running — `curl http://localhost:8001/health` should return `{"status":"ok"}`
2. You started it with `--port 8001` (not the default 8000)
</details>

<details>
<summary><b>"Stream idle timeout" on Model Performance page</b></summary>

`/models/metrics` reads all evaluation JSONs and loads figure lists from disk — it's slow on first load. The dashboard client gives it 45 seconds via `_HEAVY_TIMEOUT`. If it still times out, check that `reports/` is populated (run `dvc repro`).
</details>

---

## 🚧 Limitations & Future Work

### Honest Limitations

- **Dataset scale** — UCI's 649 rows are regional (Portuguese high schools, 2008). Models are inherently bounded by this sample.
- **Simulated behavioral stream** — The weekly attendance / quiz / LMS data is synthesized with realistic noise from G3, not scraped from a real LMS. This is disclosed transparently.
- **Rule-based recommendations** — Intervention actions are mapped via YAML. A production system would train a policy network on actual intervention outcome data.
- **No fairness audit** — Bias analysis across `sex`, `school`, `address` is not implemented. Documented as future work.
- **Single-node deployment** — SQLite + in-process threading works for ≤ thousands of students. Scale beyond that needs PostgreSQL + Kafka + Redis.

### Future Roadmap

- 🔒 Authentication & role-based access (advisor / admin / student)
- 📊 Fairness and bias audit across demographic slices
- 🔁 Policy learning from advisor action outcomes (reinforcement learning)
- ☁️ Containerization (Docker + docker-compose) and cloud deployment (AWS / GCP)
- 📧 Email / SMS alert delivery (SendGrid, Twilio)
- 🔌 LTI integration for real Canvas / Moodle LMS hookup
- 🧪 Shadow-mode A/B testing of model versions
- 📱 Native mobile app for advisors

---

## 🙏 Acknowledgements

- **[UCI ML Repository](https://archive.ics.uci.edu/)** — for the Student Performance dataset (Cortez & Silva, 2008)
- **[Anthropic](https://www.anthropic.com/)** — for Claude Haiku 4.5 and the SDK that powers the AI Advisor
- **[Facebook AI Research](https://github.com/facebookresearch/faiss)** — for FAISS
- **[Hugging Face](https://huggingface.co/sentence-transformers)** — for sentence-transformers
- **The open-source Python ML ecosystem** — scikit-learn, TensorFlow, pandas, NumPy, FastAPI, Streamlit, Plotly, SQLAlchemy, DVC

---

<div align="center">

**Built with care as a final capstone project.**  
*From raw data to live AI-assisted advisor workflow — every layer implemented in one coherent system.*

⭐ If this project helped you, consider giving it a star.

</div>
