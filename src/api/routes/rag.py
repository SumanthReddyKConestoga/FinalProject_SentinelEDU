"""RAG query and student recommendation endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.rag import rag_engine

router = APIRouter(prefix="/rag", tags=["rag"])


class QueryRequest(BaseModel):
    question: str
    student_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    student_context: str


@router.post("/query", response_model=QueryResponse)
def rag_query(body: QueryRequest):
    """Answer an advisor question, optionally grounded in a student's profile."""
    profile = None
    if body.student_id:
        from src.db.models import SessionLocal, Student, WeeklyRecord, Alert
        from sqlalchemy.orm import joinedload
        session = SessionLocal()
        try:
            student = session.query(Student).filter(
                Student.id == body.student_id
            ).first()
            if not student:
                raise HTTPException(status_code=404, detail="Student not found")

            weekly = (
                session.query(WeeklyRecord)
                .filter(WeeklyRecord.student_id == body.student_id)
                .order_by(WeeklyRecord.week)
                .all()
            )
            alerts = (
                session.query(Alert)
                .filter(Alert.student_id == body.student_id)
                .order_by(Alert.triggered_at.desc())
                .limit(10)
                .all()
            )

            profile = {
                "student": {
                    "id": student.id,
                    "current_risk_class": student.current_risk_class,
                    "current_predicted_g3": student.current_predicted_g3,
                    "segment": student.segment,
                    "school": student.school,
                    "age": student.age,
                    "sex": student.sex,
                    "static_features": student.static_features,
                },
                "weekly_records": [
                    {
                        "week": w.week,
                        "attendance_pct": w.attendance_pct,
                        "quiz_score": w.quiz_score,
                        "late_count": w.late_count,
                        "lms_logins": w.lms_logins,
                    }
                    for w in weekly
                ],
                "alerts": [
                    {
                        "rule_id": a.rule_id,
                        "severity": a.severity,
                        "description": a.description,
                        "acknowledged": a.acknowledged,
                    }
                    for a in alerts
                ],
            }
        finally:
            session.close()

    result = rag_engine.query(body.question, student_profile=profile)
    return result


@router.get("/recommend/{student_id}", response_model=QueryResponse)
def rag_recommend(student_id: str):
    """Generate AI-powered recommendations for a specific student."""
    from src.db.models import SessionLocal, Student, WeeklyRecord, Alert
    session = SessionLocal()
    try:
        student = session.query(Student).filter(Student.id == student_id).first()
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")

        weekly = (
            session.query(WeeklyRecord)
            .filter(WeeklyRecord.student_id == student_id)
            .order_by(WeeklyRecord.week)
            .all()
        )
        alerts = (
            session.query(Alert)
            .filter(Alert.student_id == student_id)
            .order_by(Alert.triggered_at.desc())
            .limit(10)
            .all()
        )

        profile = {
            "student": {
                "id": student.id,
                "current_risk_class": student.current_risk_class,
                "current_predicted_g3": student.current_predicted_g3,
                "segment": student.segment,
                "school": student.school,
                "age": student.age,
                "sex": student.sex,
                "static_features": student.static_features,
            },
            "weekly_records": [
                {
                    "week": w.week,
                    "attendance_pct": w.attendance_pct,
                    "quiz_score": w.quiz_score,
                    "late_count": w.late_count,
                    "lms_logins": w.lms_logins,
                }
                for w in weekly
            ],
            "alerts": [
                {
                    "rule_id": a.rule_id,
                    "severity": a.severity,
                    "description": a.description,
                    "acknowledged": a.acknowledged,
                }
                for a in alerts
            ],
        }
    finally:
        session.close()

    return rag_engine.recommend_for_student(profile)
