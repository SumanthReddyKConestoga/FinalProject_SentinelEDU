"""Alert routes."""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.db.models import Alert
from src.db.session import get_session

router = APIRouter(prefix="/alerts", tags=["alerts"])


class AlertOut(BaseModel):
    id: int
    student_id: str
    rule_id: str
    severity: str
    description: str
    evidence: Optional[dict]
    acknowledged: bool
    triggered_at: Optional[datetime]

    class Config:
        from_attributes = True


@router.get("/recent", response_model=List[AlertOut])
def recent_alerts(
    limit: int = Query(20, ge=1, le=200),
    session: Session = Depends(get_session),
):
    return (
        session.query(Alert)
        .order_by(Alert.triggered_at.desc())
        .limit(limit)
        .all()
    )


@router.post("/{alert_id}/acknowledge")
def acknowledge(alert_id: int, session: Session = Depends(get_session)):
    alert = session.query(Alert).filter_by(id=alert_id).first()
    if alert:
        alert.acknowledged = True
        session.commit()
    return {"status": "ok"}
