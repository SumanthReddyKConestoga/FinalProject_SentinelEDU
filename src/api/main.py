"""FastAPI application entry point."""
from dotenv import load_dotenv
load_dotenv()  # loads .env from the project root before anything else

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.db.models import init_db
from src.api.routes import students, alerts, recommendations, streaming, predictions, segments, models, rag

app = FastAPI(title="SentinelEDU API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(students.router)
app.include_router(alerts.router)
app.include_router(recommendations.router)
app.include_router(streaming.router)
app.include_router(predictions.router)
app.include_router(segments.router)
app.include_router(models.router)
app.include_router(rag.router)


@app.on_event("startup")
def startup():
    init_db()
    # Warm up inference service (loads all models)
    from src.inference.service import get_service
    get_service()
    # Hydrate feature store from DB
    from src.streaming.feature_store import get_store
    from src.db.models import SessionLocal
    session = SessionLocal()
    try:
        get_store().hydrate_from_db(session)
    finally:
        session.close()
    # Build RAG index (fast after first run — loads from disk)
    from src.rag import retriever as rag_retriever
    rag_retriever.build_index()


@app.get("/health")
def health():
    return {"status": "ok"}
