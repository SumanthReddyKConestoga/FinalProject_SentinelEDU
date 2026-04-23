"""Small helpers."""
from pathlib import Path
import json
from datetime import datetime


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_json(path: Path) -> dict:
    with open(Path(path), "r") as f:
        return json.load(f)


def now_iso() -> str:
    return datetime.utcnow().isoformat()
