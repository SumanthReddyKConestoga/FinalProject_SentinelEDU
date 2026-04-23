# 🚧 CONTINUE HERE — Instructions for Claude Code

This project was scaffolded in a separate session. The **DVC pipeline, all ML/DL models, preprocessing, and database layer are complete and working**. Your job is to finish the remaining integration + UI layers.

---

## Step 1: Verify the Foundation Works

Run these commands first. If any fails, fix it before proceeding.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize DVC (skip if .dvc already exists)
dvc init --no-scm

# 3. Run the full pipeline — loads data, preprocesses, trains all models
dvc repro

# 4. Seed the SQLite DB
python -m src.db.seed
```

Expected outcome after `dvc repro`:
- `data/raw/student_performance.csv` exists
- `data/processed/*.parquet` files exist
- `data/streaming/events.jsonl` exists
- `artifacts/preprocessor.pkl` exists
- `models/regression/*.pkl`, `models/classification/*.pkl`, `models/deep/*.keras`, `models/cnn/cnn1d.keras`, `models/clustering/kmeans.pkl` all exist
- `reports/figures/*.png` — confusion matrices, loss curves, residual plots
- `reports/model_comparison.md` summarizes all models

After `python -m src.db.seed`:
- `sentinel.db` exists in project root with students, weekly_records, predictions populated

---

## Step 2: Build the Remaining Modules

Build these **in order**. Every module should follow the coding style of what's already in `src/` — small classes, clear logging via `src.utils.logging.get_logger`, config loaded via `src.config.SETTINGS`/`THRESHOLDS`.

### 2a. `src/inference/` — Inference Service

**Files to create:**
- `src/inference/registry.py` — reads `config/model_registry.json`, loads the production models (preprocessor + regression + classifier + cnn + kmeans) at startup. Handles both joblib `.pkl` and Keras `.keras` files. Keras classifier may live under `models/deep/` if the best classifier was a deep one.
- `src/inference/service.py` — `InferenceService` class with methods:
  - `predict_regression(feature_row: dict) -> float`
  - `predict_classification(feature_row: dict) -> (label: str, confidence: float)`
  - `predict_cnn(sequence: np.ndarray) -> (label: str, confidence: float)` where sequence is shape `(window, n_features)`
  - `assign_segment(feature_row: dict) -> str` using KMeans
  - Single `feature_row` goes through `Preprocessor.transform()` first. Thread-safe after startup.

### 2b. `src/streaming/` — Streaming Producer + Consumer

**Files to create:**
- `src/streaming/feature_store.py` — `FeatureStore` class with per-student `collections.deque(maxlen=8)` rolling windows in memory. Hydrates from `weekly_records` table on startup. Methods: `update(student_id, event)`, `get_window(student_id) -> np.ndarray or None`, `compute_derived(student_id) -> dict` (returns trend slopes, means, latest values).
- `src/streaming/producer.py` — `StreamProducer` runs in a `threading.Thread`, reads `data/streaming/events.jsonl` line by line, pushes events into a shared `queue.Queue` at rate from `SETTINGS["streaming"]["rate_events_per_sec"]`. Supports start/stop/loop.
- `src/streaming/consumer.py` — `StreamConsumer` also in a thread. Pulls from queue → `FeatureStore.update` → `InferenceService` → write `Prediction` row → `AlertEngine.evaluate` → if alert fires, write `Alert` + call `RecommendationEngine.generate` → write `Recommendation`. Each event writes a `WeeklyRecord` too so the dashboard sees live updates.
- Module-level singletons for producer + consumer + shared queue so the API can control them.

### 2c. `src/alerts/` — Alert Engine

**Files:**
- `src/alerts/rules.py` — `evaluate_rule(rule_dict, feature_map) -> bool`. Supports three rule types from `config/thresholds.yaml`:
  - `threshold`: `feature operator value` — operators `lt, lte, gt, gte, eq, neq`
  - `slope`: same but `feature` is a computed slope (e.g. `quiz_score_slope_3`)
  - `compound`: list of sub-conditions ANDed together
- `src/alerts/engine.py` — `AlertEngine.evaluate(student_id, feature_map, db_session)`:
  - Loads rules from `THRESHOLDS["rules"]`
  - For each rule that fires, check cooldown (no duplicate same-rule alert within `cooldown_hours` unless severity escalated)
  - Write `Alert` rows to DB
  - Return list of new alerts

Feature map passed in should include: `weekly_attendance_pct`, `weekly_quiz_score`, `weekly_late_count`, `quiz_score_slope_3` (computed from last 3 weeks), `regression_prediction`, `classifier_label`, `classifier_confidence`, `cnn_high_risk_consecutive` (count of consecutive windows CNN said "High").

### 2d. `src/recommendations/` — Recommendation Engine

**Files:**
- `src/recommendations/engine.py` — `RecommendationEngine.generate(alert, student, db_session)`:
  - Reads `THRESHOLDS["recommendations"][rule_id]["actions"]`
  - For each action, check intervention fatigue (not recommended within 14 days already): skip if fatigued
  - Write `Recommendation` rows with priority from YAML
  - Apply a severity multiplier: critical=1.0, high=0.85, medium=0.6, low=0.3
  - Return list of new recommendations sorted by priority desc

### 2e. `src/api/` — FastAPI Backend

**`src/api/main.py`:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import students, alerts, recommendations, streaming, predictions
from src.db.models import init_db

app = FastAPI(title="SentinelEDU API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(students.router)
app.include_router(alerts.router)
app.include_router(recommendations.router)
app.include_router(streaming.router)
app.include_router(predictions.router)

@app.on_event("startup")
def startup():
    init_db()
    # Load inference service singleton

@app.get("/health")
def health(): return {"status": "ok"}
```

**Routes to implement:**
- `GET /students` → list w/ pagination + optional `risk_class` filter
- `GET /students/{id}` → single student row
- `GET /students/{id}/profile` → student + latest predictions + weekly records + alerts + recommendations (everything the Profile page needs)
- `GET /students/{id}/weekly` → weekly records (for trend chart)
- `GET /students/{id}/alerts` → alert history
- `GET /students/{id}/recommendations` → current recs
- `GET /alerts/recent?limit=20` → feed for Home page
- `POST /alerts/{id}/acknowledge` → mark acknowledged
- `POST /students/{id}/advisor-action` → log an AdvisorAction
- `POST /streaming/start` → start producer + consumer threads
- `POST /streaming/stop` → stop them
- `GET /streaming/status` → running bool, queue depth, events processed
- `GET /segments` → segment summary with counts
- `GET /models/metrics` → reads and returns `reports/*.json` contents for Model Performance page

Use Pydantic response models. Dependency injection with `Depends(get_session)` from `src.db.session`.

### 2f. `dashboard/` — Streamlit Multi-Page App

**Theme — `dashboard/theme/custom.css`:**

Premium look. Use these design tokens:
- Fonts: Inter or system-ui. Large weight contrast.
- Risk colors: `--low:#10B981; --medium:#F59E0B; --high:#EF4444; --critical:#7C2D12;`
- Neutral: `--bg:#F8FAFC; --card:#FFFFFF; --text:#0F172A; --muted:#64748B;`
- Cards: 12px radius, subtle shadow, 24px padding.
- Generous spacing (8px base unit, 24px gutters).
- Replace Streamlit's default button styling with a clean pill.

Inject via:
```python
with open("dashboard/theme/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
```

**`dashboard/api_client.py`** — thin `requests` wrapper around all API endpoints. Single `BASE_URL = "http://localhost:8000"`. Wrap each call in try/except and return dicts or None.

**`dashboard/app.py`** — entry point. Sets page config (`layout="wide"`), loads custom CSS, provides a sidebar with navigation + a global "Stream Status" indicator (green dot when running) + Start/Stop Stream buttons that hit `/streaming/start` and `/streaming/stop`.

**Pages:**

**`dashboard/pages/1_🏠_Home.py`** — "Mission Control":
- Header + description
- 4 KPI cards in a row: Total Students, Active Alerts, High-Risk Count, Interventions This Week. Each uses colored accent bar.
- Two-column layout:
  - LEFT (60%): Live alert timeline. Use `st.rerun()` or `streamlit-autorefresh` every 3 seconds. Each alert rendered as a card with colored left border matching severity, rule description, student name (clickable → navigates to Student Profile), timestamp.
  - RIGHT (40%): Plotly donut chart of risk distribution (Low/Medium/High counts). Below it, a Plotly bar chart of segment distribution.

**`dashboard/pages/2_👥_Roster.py`** — Student Roster:
- Filter bar: search by name/id, filter by risk_class, filter by segment.
- Sortable Pandas table with columns: ID, Sex, Risk, Predicted G3, Segment, Active Alerts count. Risk column colored via conditional formatting (use `st.dataframe(df.style.apply(...))`).
- Each row has a "View" button → switches to Profile page with `st.session_state["selected_student"] = sid`.

**`dashboard/pages/3_👤_Student_Profile.py`** — Deep dive:
Use `st.session_state.get("selected_student")` or a text input for student ID.
- Header: name/ID, risk badge, program, predicted G3 with confidence
- Row of small metric cards: Attendance %, Avg Quiz Score, Late Submissions, LMS Logins
- Plotly multi-line chart: attendance / quiz_score / submission_rate over weeks (dual y-axis or normalized)
- Plotly chart of predicted G3 over time (from Prediction rows)
- Active Alerts section: list of cards with severity colors
- Recommendations section: ranked cards with "Take Action" button that POSTs an AdvisorAction
- Segment info + behavioral summary

**`dashboard/pages/4_🚨_Alerts.py`** — Alert Center:
- Global alert timeline with filters (severity, date range, acknowledged/unacknowledged)
- Bulk acknowledge action

**`dashboard/pages/5_📊_Model_Performance.py`** — Evaluation:
- Tabs: Regression / Classification / Deep Learning / CNN / Clustering
- Each tab shows metrics table + relevant PNG from `reports/figures/`
- Markdown render of `reports/model_comparison.md` at the bottom

**Shared Components** (`dashboard/components/`):
- `cards.py` — `kpi_card(label, value, delta, color)`, `alert_card(alert)`, `recommendation_card(rec, on_action_callback)`
- `charts.py` — `risk_donut(counts)`, `trend_chart(weekly_df)`, `prediction_timeline(predictions_df)`
- `tables.py` — styled dataframes

---

## Step 3: Integration Testing

1. Start backend: `uvicorn src.api.main:app --reload --port 8000`
2. Verify `http://localhost:8000/docs` loads
3. Start dashboard: `streamlit run dashboard/app.py`
4. Home page should load with seeded data (alerts timeline empty initially)
5. Click "Start Stream" in sidebar
6. Alerts should start appearing within 10–20 seconds as the producer emits events and thresholds trigger
7. Click into a student with a red badge → see trend chart + active alerts + recommendations

---

## Step 4: Demo Prep

- Make sure at least 3 students have dramatic declining trends so alerts fire visibly. If not, tune thresholds in `config/thresholds.yaml`.
- Screen-record a 3-minute run as backup.
- `reports/model_comparison.md` is your "results" slide — lift straight from it.

---

## Troubleshooting

- **TensorFlow install fails:** try `pip install tensorflow-cpu==2.17.0` instead.
- **DVC stage fails:** run stages individually: `python -m src.training.stage_load_data`, then `stage_preprocess`, etc. Catch the real error.
- **CNN gets 0% accuracy:** check sequence shape is `(n_samples, 8, 5)` and labels are 0/1/2 integers.
- **Streaming doesn't show alerts:** lower the thresholds in `thresholds.yaml` temporarily (e.g. attendance < 85 instead of 70) to force triggers.
- **Streamlit autorefresh:** if `streamlit-autorefresh` isn't installed, use `time.sleep(3); st.rerun()` at the end of Home page.

---

## Design Principle Reminders

- Every screen should feel calm until something goes wrong.
- Risk colors are always paired with icon + label for accessibility.
- Keep prose short, use verbs on buttons ("Schedule Meeting" not "Meeting").
- Use Plotly theme `plotly_white` everywhere for consistency.
- No clutter — generous whitespace, clear hierarchy.

Good luck. You've got ~10–15 hours. The ML side is already done; the remaining work is integration + UI polish. Focus on getting the end-to-end flow working first, then beautify.
