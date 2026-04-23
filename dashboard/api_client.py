"""Thin requests wrapper around the SentinelEDU API."""
import requests

BASE_URL = "http://localhost:8001"
_TIMEOUT = 15        # default — enough for profile/roster endpoints
_HEAVY_TIMEOUT = 45  # model metrics and similar slow endpoints


def _get(
    path: str, params: dict = None, timeout: int = _TIMEOUT
) -> dict | list | None:
    try:
        r = requests.get(f"{BASE_URL}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _post(path: str, json: dict = None) -> dict | None:
    try:
        r = requests.post(f"{BASE_URL}{path}", json=json, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# Health
def health() -> dict | None:
    return _get("/health")


# Students
def list_students(risk_class: str = None, limit: int = 500) -> list:
    params = {"limit": limit}
    if risk_class:
        params["risk_class"] = risk_class
    return _get("/students", params) or []


def get_student(student_id: str) -> dict | None:
    return _get(f"/students/{student_id}")


def get_profile(student_id: str) -> dict | None:
    return _get(f"/students/{student_id}/profile")


def get_weekly(student_id: str) -> list:
    return _get(f"/students/{student_id}/weekly") or []


def get_student_alerts(student_id: str) -> list:
    return _get(f"/students/{student_id}/alerts") or []


def get_student_recommendations(student_id: str) -> list:
    return _get(f"/students/{student_id}/recommendations") or []


def log_advisor_action(
    student_id: str, action_taken: str,
    rec_id: int = None, notes: str = "",
) -> dict | None:
    return _post(f"/students/{student_id}/advisor-action", {
        "action_taken": action_taken,
        "recommendation_id": rec_id,
        "notes": notes,
    })


# Alerts
def recent_alerts(limit: int = 30) -> list:
    return _get("/alerts/recent", {"limit": limit}) or []


def acknowledge_alert(alert_id: int) -> dict | None:
    return _post(f"/alerts/{alert_id}/acknowledge")


# Streaming
def start_stream() -> dict | None:
    return _post("/streaming/start")


def stop_stream() -> dict | None:
    return _post("/streaming/stop")


def stream_status() -> dict | None:
    return _get("/streaming/status")


# Segments
def segment_summary() -> list:
    return _get("/segments") or []


# Models — these endpoints read from disk so we give them extra time
def model_metrics() -> dict:
    return _get("/models/metrics", timeout=_HEAVY_TIMEOUT) or {}


def model_registry() -> dict:
    return _get("/models/registry", timeout=_HEAVY_TIMEOUT) or {}


def model_comparison() -> str:
    data = _get("/models/comparison", timeout=_HEAVY_TIMEOUT)
    return data.get("markdown", "") if data else ""


def list_figures() -> list:
    return _get("/models/figures", timeout=_HEAVY_TIMEOUT) or []


# RAG
def rag_query(question: str, student_id: str = None) -> dict | None:
    payload = {"question": question}
    if student_id:
        payload["student_id"] = student_id
    return _post("/rag/query", payload)


def rag_recommend(student_id: str) -> dict | None:
    return _get(f"/rag/recommend/{student_id}", timeout=_HEAVY_TIMEOUT)
