"""Segment summary routes."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.db.models import Student
from src.db.session import get_session

router = APIRouter(prefix="/segments", tags=["segments"])


@router.get("")
def segment_summary(session: Session = Depends(get_session)):
    students = session.query(Student).all()
    counts: dict = {}
    for s in students:
        seg = s.segment or "Unknown"
        counts[seg] = counts.get(seg, 0) + 1
    return [{"segment": k, "count": v} for k, v in sorted(counts.items())]
