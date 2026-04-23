"""Prediction routes."""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.models import Prediction
from src.db.session import get_session

router = APIRouter(prefix="/predictions", tags=["predictions"])


class PredictionOut(BaseModel):
    id: int
    student_id: str
    model_name: str
    prediction_value: Optional[float]
    predicted_class: Optional[str]
    confidence: Optional[float]
    week: Optional[int]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


@router.get("", response_model=List[PredictionOut])
def list_predictions(
    student_id: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_session),
):
    q = session.query(Prediction).order_by(Prediction.created_at.desc())
    if student_id:
        q = q.filter(Prediction.student_id == student_id)
    if model_name:
        q = q.filter(Prediction.model_name == model_name)
    return q.limit(limit).all()
