"""AlertEngine — evaluates rules, respects cooldowns, writes Alert rows."""
from datetime import datetime, timedelta
from typing import List

from src.config import THRESHOLDS
from src.alerts.rules import evaluate_rule
from src.db.models import Alert
from src.utils.logging import get_logger

log = get_logger(__name__)

_SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class AlertEngine:
    def evaluate(self, student_id: str, feature_map: dict, session) -> List[Alert]:
        """Evaluate all rules for a student; write and return new Alert rows."""
        rules = THRESHOLDS.get("rules", [])
        new_alerts: List[Alert] = []

        for rule in rules:
            if not evaluate_rule(rule, feature_map):
                continue

            rule_id = rule["id"]
            severity = rule.get("severity", "low")
            cooldown_h = rule.get("cooldown_hours", 24)
            cooldown_cutoff = datetime.utcnow() - timedelta(hours=cooldown_h)

            # Check existing alert within cooldown window
            existing = (
                session.query(Alert)
                .filter_by(student_id=student_id, rule_id=rule_id)
                .filter(Alert.triggered_at >= cooldown_cutoff)
                .order_by(Alert.triggered_at.desc())
                .first()
            )
            if existing:
                # Allow escalation: only skip if same or higher severity already recorded
                existing_sev = _SEVERITY_ORDER.get(existing.severity, 0)
                new_sev = _SEVERITY_ORDER.get(severity, 0)
                if existing_sev >= new_sev:
                    continue

            evidence = {
                k: v for k, v in feature_map.items()
                if isinstance(v, (int, float, str, bool))
            }
            alert = Alert(
                student_id=student_id,
                rule_id=rule_id,
                severity=severity,
                description=rule.get("description", rule_id),
                evidence=evidence,
                acknowledged=False,
                triggered_at=datetime.utcnow(),
            )
            session.add(alert)
            session.flush()  # get alert.id populated
            new_alerts.append(alert)
            log.info(f"Alert fired: student={student_id}, rule={rule_id}, severity={severity}")

        return new_alerts
