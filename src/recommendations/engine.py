"""RecommendationEngine — maps alerts to ranked advisor actions."""
from datetime import datetime, timedelta
from typing import List

from src.config import THRESHOLDS
from src.db.models import Alert, Recommendation, Student
from src.utils.logging import get_logger

log = get_logger(__name__)

_SEVERITY_MULTIPLIER = {
    "critical": 1.00,
    "high": 0.85,
    "medium": 0.60,
    "low": 0.30,
}
_FATIGUE_DAYS = 14


class RecommendationEngine:
    def generate(self, alert: Alert, student: Student, session) -> List[Recommendation]:
        """Generate recommendation rows for a triggered alert."""
        rule_recs = THRESHOLDS.get("recommendations", {}).get(alert.rule_id, {})
        actions = rule_recs.get("actions", [])
        if not actions:
            return []

        multiplier = _SEVERITY_MULTIPLIER.get(alert.severity, 0.5)
        fatigue_cutoff = datetime.utcnow() - timedelta(days=_FATIGUE_DAYS)
        new_recs: List[Recommendation] = []

        for action_def in actions:
            action = action_def["action"]
            base_priority = float(action_def.get("priority", 0.5))
            rationale = action_def.get("rationale", "")

            # Intervention fatigue check
            recent = (
                session.query(Recommendation)
                .filter_by(student_id=student.id, action=action)
                .filter(Recommendation.created_at >= fatigue_cutoff)
                .first()
            )
            if recent:
                log.debug(f"Fatigue skip: student={student.id}, action={action}")
                continue

            priority = round(base_priority * multiplier, 4)
            rec = Recommendation(
                student_id=student.id,
                alert_id=alert.id,
                action=action,
                rationale=rationale,
                priority=priority,
                status="pending",
                created_at=datetime.utcnow(),
            )
            session.add(rec)
            new_recs.append(rec)
            log.info(f"Recommendation: student={student.id}, action={action}, priority={priority:.2f}")

        new_recs.sort(key=lambda r: r.priority, reverse=True)
        return new_recs
