"""Model metrics and registry routes."""
import os
from pathlib import Path

from fastapi import APIRouter

from src.config import resolve_path, SETTINGS
from src.utils.helpers import read_json

router = APIRouter(prefix="/models", tags=["models"])

REPORTS = resolve_path(SETTINGS["paths"]["reports"])


@router.get("/metrics")
def model_metrics():
    result = {}
    for fname in [
        "regression_metrics.json",
        "classification_metrics.json",
        "clustering_metrics.json",
        "deep_metrics.json",
        "cnn_metrics.json",
        "regression_cv.json",
    ]:
        path = REPORTS / fname
        if path.exists():
            key = fname.replace(".json", "")
            result[key] = read_json(path)
    return result


@router.get("/registry")
def model_registry():
    path = resolve_path("config/model_registry.json")
    return read_json(path)


@router.get("/comparison")
def model_comparison():
    path = REPORTS / "model_comparison.md"
    if path.exists():
        return {"markdown": path.read_text()}
    return {"markdown": "No comparison report found."}


@router.get("/figures")
def list_figures():
    fig_dir = REPORTS / "figures"
    if not fig_dir.exists():
        return []
    return [f.name for f in fig_dir.iterdir() if f.suffix == ".png"]
