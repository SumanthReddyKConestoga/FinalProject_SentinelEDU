"""Alert rule evaluation — threshold, slope, and compound types."""
from typing import Any

from src.utils.logging import get_logger

log = get_logger(__name__)

_OPERATORS = {
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "eq": lambda a, b: str(a) == str(b),
    "neq": lambda a, b: str(a) != str(b),
}


def _check_condition(feature_map: dict, feature: str, operator: str, value: Any) -> bool:
    feat_val = feature_map.get(feature)
    if feat_val is None:
        return False
    op_fn = _OPERATORS.get(operator)
    if op_fn is None:
        log.warning(f"Unknown operator: {operator}")
        return False
    try:
        return op_fn(feat_val, value)
    except Exception as exc:
        log.warning(f"Rule evaluation error (feature={feature}): {exc}")
        return False


def evaluate_rule(rule: dict, feature_map: dict) -> bool:
    """Return True if the rule fires given the feature_map."""
    ctype = rule.get("condition_type", "threshold")

    if ctype in ("threshold", "slope"):
        return _check_condition(
            feature_map,
            rule["feature"],
            rule["operator"],
            rule["value"],
        )

    if ctype == "compound":
        conditions = rule.get("conditions", [])
        return all(
            _check_condition(feature_map, c["feature"], c["operator"], c["value"])
            for c in conditions
        )

    log.warning(f"Unknown condition_type: {ctype}")
    return False
