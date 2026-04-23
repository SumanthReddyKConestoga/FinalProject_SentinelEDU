"""Configuration loader for SentinelEDU.

Loads YAML settings and thresholds once, exposes them as a singleton.
"""
from pathlib import Path
from functools import lru_cache
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


@lru_cache(maxsize=1)
def load_settings() -> dict:
    """Load main application settings."""
    with open(CONFIG_DIR / "settings.yaml", "r") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_thresholds() -> dict:
    """Load alert and recommendation rules."""
    with open(CONFIG_DIR / "thresholds.yaml", "r") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    return PROJECT_ROOT


def resolve_path(relative: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / relative


# Convenience exports
SETTINGS = load_settings()
THRESHOLDS = load_thresholds()
