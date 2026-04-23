"""
Unified RAG engine — single entry point for the API and dashboard.
"""
from __future__ import annotations
import logging

from src.rag import retriever, generator

logger = logging.getLogger(__name__)

# Build the index once at module import (fast after first run — loads from disk)
try:
    retriever._ensure_loaded()
except Exception as exc:
    logger.warning("RAG index warm-up failed: %s", exc)


def _format_student_context(profile: dict) -> str:
    """Convert a student profile dict into a plain-text context string."""
    if not profile:
        return "No student data provided."

    student = profile.get("student", {})
    sf      = student.get("static_features") or {}

    weekly = profile.get("weekly_records", [])
    if weekly:
        latest = weekly[-1]
        att    = latest.get("attendance_pct", "N/A")
        quiz   = latest.get("quiz_score", "N/A")
        late   = latest.get("late_count", "N/A")
        logins = latest.get("lms_logins", "N/A")
    else:
        att = sf.get("mean_attendance_pct", "N/A")
        quiz = sf.get("mean_quiz_score", "N/A")
        late = "N/A"
        logins = "N/A"

    alerts = profile.get("alerts", [])
    active_alerts = [a for a in alerts if not a.get("acknowledged")]
    alert_summary = (
        ", ".join(f"{a['severity']} ({a['rule_id']})" for a in active_alerts[:5])
        if active_alerts else "None"
    )

    return (
        f"Student ID: {student.get('id', 'N/A')}\n"
        f"Risk Class: {student.get('current_risk_class', 'N/A')}\n"
        f"Predicted Final Grade (G3): {student.get('current_predicted_g3', 'N/A')}\n"
        f"Segment: {student.get('segment', 'N/A')}\n"
        f"School: {student.get('school', 'N/A')}  Age: {student.get('age', 'N/A')}  "
        f"Sex: {student.get('sex', 'N/A')}\n"
        f"Latest Attendance: {att}%\n"
        f"Latest Quiz Score: {quiz}\n"
        f"Late Submissions (latest week): {late}\n"
        f"LMS Logins (latest week): {logins}\n"
        f"Active Alerts: {alert_summary}\n"
    )


def query(
    user_question: str,
    student_profile: dict | None = None,
    top_k: int = 3,
) -> dict:
    """
    Main RAG query entry point.

    Returns:
        {
          "answer": str,
          "sources": [{"title": str, "category": str}, ...],
          "student_context": str,
        }
    """
    student_ctx = _format_student_context(student_profile or {})

    # Augment the retrieval query with student risk context for better relevance
    risk = ""
    if student_profile:
        risk = student_profile.get("student", {}).get("current_risk_class", "")
    retrieval_query = f"{user_question} {risk} student".strip()

    docs = retriever.retrieve(retrieval_query, top_k=top_k)

    answer = generator.generate(
        query=user_question,
        context_docs=docs,
        student_context=student_ctx,
    )

    sources = [{"title": d["title"], "category": d["category"]} for d in docs]

    return {
        "answer": answer,
        "sources": sources,
        "student_context": student_ctx,
    }


def recommend_for_student(student_profile: dict) -> dict:
    """
    Generate AI-powered recommendations for a specific student profile.
    Wraps query() with a pre-built prompt based on the student's risk indicators.
    """
    student = student_profile.get("student", {})
    risk    = student.get("current_risk_class", "Unknown")
    sid     = student.get("id", "Unknown")

    alerts  = student_profile.get("alerts", [])
    active  = [a for a in alerts if not a.get("acknowledged")]
    alert_descriptions = "; ".join(a.get("description", "") for a in active[:3])

    question = (
        f"Student {sid} is classified as {risk} risk. "
        f"Active alerts: {alert_descriptions or 'none'}. "
        "What are the most important interventions an advisor should take right now, "
        "and what should they say in the first conversation with this student?"
    )

    return query(question, student_profile=student_profile)
