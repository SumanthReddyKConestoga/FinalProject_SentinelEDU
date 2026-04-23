"""DB session dependency for FastAPI + other consumers."""
from src.db.models import SessionLocal


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
