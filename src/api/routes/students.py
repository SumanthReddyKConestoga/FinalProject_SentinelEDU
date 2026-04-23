"""Student routes."""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.models import Student, Prediction, WeeklyRecord, Alert, Recommendation, AdvisorAction
from src.db.session import get_session

router = APIRouter(prefix="/students", tags=["students"])


class StudentOut(BaseModel):
    id: str
    age: Optional[int]
    sex: Optional[str]
    school: Optional[str]
    program: Optional[str]
    segment: Optional[str]
    current_risk_class: Optional[str]
    current_predicted_g3: Optional[float]

    class Config:
        from_attributes = True


class WeeklyOut(BaseModel):
    id: int
    week: int
    attendance_pct: Optional[float]
    quiz_score: Optional[float]
    submission_rate: Optional[float]
    lms_logins: Optional[int]
    late_count: Optional[int]

    class Config:
        from_attributes = True


class PredictionOut(BaseModel):
    id: int
    model_name: str
    prediction_value: Optional[float]
    predicted_class: Optional[str]
    confidence: Optional[float]
    week: Optional[int]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class AlertOut(BaseModel):
    id: int
    rule_id: str
    severity: str
    description: str
    evidence: Optional[dict]
    acknowledged: bool
    triggered_at: Optional[datetime]

    class Config:
        from_attributes = True


class RecommendationOut(BaseModel):
    id: int
    alert_id: Optional[int]
    action: str
    rationale: str
    priority: float
    status: str
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class ProfileOut(BaseModel):
    student: StudentOut
    latest_predictions: List[PredictionOut]
    weekly_records: List[WeeklyOut]
    alerts: List[AlertOut]
    recommendations: List[RecommendationOut]


class AdvisorActionIn(BaseModel):
    advisor_name: str = "Advisor"
    action_taken: str
    recommendation_id: Optional[int] = None
    notes: str = ""


@router.get("", response_model=List[StudentOut])
def list_students(
    risk_class: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    q = session.query(Student)
    if risk_class:
        q = q.filter(Student.current_risk_class == risk_class)
    return q.offset(offset).limit(limit).all()


@router.get("/{student_id}", response_model=StudentOut)
def get_student(student_id: str, session: Session = Depends(get_session)):
    student = session.query(Student).filter_by(id=student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


@router.get("/{student_id}/profile", response_model=ProfileOut)
def get_profile(student_id: str, session: Session = Depends(get_session)):
    student = session.query(Student).filter_by(id=student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    preds = (
        session.query(Prediction)
        .filter_by(student_id=student_id)
        .order_by(Prediction.created_at.desc())
        .limit(20)
        .all()
    )
    weekly = (
        session.query(WeeklyRecord)
        .filter_by(student_id=student_id)
        .order_by(WeeklyRecord.week)
        .all()
    )
    alerts = (
        session.query(Alert)
        .filter_by(student_id=student_id)
        .order_by(Alert.triggered_at.desc())
        .limit(50)
        .all()
    )
    recs = (
        session.query(Recommendation)
        .filter_by(student_id=student_id)
        .order_by(Recommendation.priority.desc())
        .limit(20)
        .all()
    )

    return ProfileOut(
        student=StudentOut.model_validate(student),
        latest_predictions=[PredictionOut.model_validate(p) for p in preds],
        weekly_records=[WeeklyOut.model_validate(w) for w in weekly],
        alerts=[AlertOut.model_validate(a) for a in alerts],
        recommendations=[RecommendationOut.model_validate(r) for r in recs],
    )


@router.get("/{student_id}/weekly", response_model=List[WeeklyOut])
def get_weekly(student_id: str, session: Session = Depends(get_session)):
    return (
        session.query(WeeklyRecord)
        .filter_by(student_id=student_id)
        .order_by(WeeklyRecord.week)
        .all()
    )


@router.get("/{student_id}/alerts", response_model=List[AlertOut])
def get_student_alerts(student_id: str, session: Session = Depends(get_session)):
    return (
        session.query(Alert)
        .filter_by(student_id=student_id)
        .order_by(Alert.triggered_at.desc())
        .limit(100)
        .all()
    )


@router.get("/{student_id}/recommendations", response_model=List[RecommendationOut])
def get_student_recommendations(student_id: str, session: Session = Depends(get_session)):
    return (
        session.query(Recommendation)
        .filter_by(student_id=student_id)
        .order_by(Recommendation.priority.desc())
        .all()
    )


@router.post("/{student_id}/advisor-action")
def log_advisor_action(
    student_id: str,
    body: AdvisorActionIn,
    session: Session = Depends(get_session),
):
    student = session.query(Student).filter_by(id=student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    action = AdvisorAction(
        student_id=student_id,
        recommendation_id=body.recommendation_id,
        advisor_name=body.advisor_name,
        action_taken=body.action_taken,
        notes=body.notes,
    )
    session.add(action)
    # Mark linked recommendation as completed
    if body.recommendation_id:
        rec = session.query(Recommendation).filter_by(id=body.recommendation_id).first()
        if rec:
            rec.status = "completed"
    session.commit()
    return {"status": "ok", "action_id": action.id}
