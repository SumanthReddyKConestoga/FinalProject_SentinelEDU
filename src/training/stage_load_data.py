"""DVC stage 1: load raw data.

Loads UCI student data (or synthetic fallback) and persists to data/raw.
"""
from src.data.loader import DataLoader
from src.utils.logging import get_logger

log = get_logger(__name__)


def main():
    loader = DataLoader()
    df = loader.load()
    log.info(f"Raw data ready: shape={df.shape}, columns={list(df.columns)[:8]}...")


if __name__ == "__main__":
    main()
