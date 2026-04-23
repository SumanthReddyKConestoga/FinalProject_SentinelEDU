"""Recommendation routes."""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.models import Recommendation
from src.db.session import get_session

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecOut(BaseModel):
    id: int
    student_id: str
    alert_id: Optional[int]
    action: str
    rationale: str
    priority: float
    status: str
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


@router.get("", response_model=List[RecOut])
def list_recommendations(
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    session: Session = Depends(get_session),
):
    q = session.query(Recommendation).order_by(Recommendation.priority.desc())
    if status:
        q = q.filter(Recommendation.status == status)
    return q.limit(limit).all()
