"""DVC stage 5: consolidated evaluation report + update model registry.

Reads all metrics JSONs, produces a model_comparison.md, and writes
the chosen production models into config/model_registry.json.
"""
from pathlib import Path
import json
from datetime import datetime

from src.config import SETTINGS, resolve_path
from src.utils.helpers import read_json, write_json, now_iso
from src.utils.logging import get_logger

log = get_logger(__name__)

REPORTS = resolve_path(SETTINGS["paths"]["reports"])
REGISTRY = resolve_path("config/model_registry.json")


def _safe_read(path: Path, default=None):
    try:
        return read_json(path)
    except Exception:
        return default if default is not None else {}


def main():
    reg_metrics = _safe_read(REPORTS / "regression_metrics.json")
    clf_metrics = _safe_read(REPORTS / "classification_metrics.json")
    cluster_metrics = _safe_read(REPORTS / "clustering_metrics.json")
    deep_metrics = _safe_read(REPORTS / "deep_metrics.json")
    cnn_metrics = _safe_read(REPORTS / "cnn_metrics.json")

    # Pick best per task
    best_regression = min(
        reg_metrics.items(), key=lambda kv: kv[1]["rmse"]
    )[0] if reg_metrics else None

    all_classifiers = {**clf_metrics, **deep_metrics}
    if cnn_metrics:
        all_classifiers["cnn1d"] = cnn_metrics
    best_classifier = max(
        all_classifiers.items(), key=lambda kv: kv[1]["f1_macro"]
    )[0] if all_classifiers else None

    registry = {
        "production": {
            "regression": best_regression,
            "classification": best_classifier,
            "cnn": "cnn1d" if cnn_metrics else None,
            "clustering": "kmeans",
        },
        "candidates": {
            "regression": list(reg_metrics.keys()),
            "classification": list(all_classifiers.keys()),
        },
        "last_updated": now_iso(),
    }
    write_json(REGISTRY, registry)

    # Build markdown report
    lines = [
        "# Model Comparison Report",
        f"_Generated: {datetime.utcnow().isoformat()}Z_",
        "",
        "## Regression (predicting G3)",
        "",
        "| Model | RMSE | MAE | R² |",
        "|---|---|---|---|",
    ]
    for name, m in reg_metrics.items():
        lines.append(
            f"| `{name}` | {m['rmse']:.3f} | {m['mae']:.3f} | {m['r2']:.3f} |"
        )

    lines += [
        "",
        "## Classification (predicting risk_class)",
        "",
        "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |",
        "|---|---|---|---|---|",
    ]
    for name, m in all_classifiers.items():
        lines.append(
            f"| `{name}` | {m['accuracy']:.3f} | {m['precision_macro']:.3f}"
            f" | {m['recall_macro']:.3f} | {m['f1_macro']:.3f} |"
        )

    lines += [
        "",
        "## Clustering",
        "",
        f"- K-Means (k=4) silhouette score: **{cluster_metrics.get('silhouette_score', 0):.3f}**",
        "",
        "## Production Choices",
        "",
        f"- Regression: **{best_regression}**",
        f"- Classification: **{best_classifier}**",
        f"- CNN: **{'cnn1d' if cnn_metrics else 'not trained'}**",
        f"- Clustering: **kmeans**",
    ]

    report_path = REPORTS / "model_comparison.md"
    report_path.write_text("\n".join(lines))
    log.info(f"Model comparison written: {report_path}")
    log.info(f"Registry updated: {REGISTRY}")


if __name__ == "__main__":
    main()
