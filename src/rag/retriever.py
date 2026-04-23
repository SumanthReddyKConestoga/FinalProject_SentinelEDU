"""
FAISS-based vector retriever with numpy cosine-similarity fallback.
Index is built once at startup and persisted to artifacts/rag_index.
"""
from __future__ import annotations
import json
import logging
import os
import numpy as np
from pathlib import Path

from src.rag.knowledge_base import get_all_documents, get_document_texts
from src.rag import embedder as emb

logger = logging.getLogger(__name__)

_INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "rag_index"
_INDEX_PATH = _INDEX_DIR / "faiss.index"
_DOCS_PATH  = _INDEX_DIR / "docs.json"

_index = None
_documents: list[dict] = []
_embeddings: np.ndarray | None = None   # fallback when FAISS not available
_use_faiss = False


def _cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between (1,D) query and (N,D) matrix."""
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return (m @ q.T).flatten()


def build_index(force: bool = False):
    """Embed all documents and build/persist the FAISS (or numpy) index."""
    global _index, _documents, _embeddings, _use_faiss

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)

    _documents = get_all_documents()
    texts = get_document_texts()
    emb.fit_tfidf(texts)

    vecs = emb.embed_texts(texts)          # (N, D) float32

    try:
        import faiss
        dim = vecs.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(vecs)
        faiss.write_index(idx, str(_INDEX_PATH))
        _index = idx
        _use_faiss = True
        logger.info("FAISS index built: %d docs, dim=%d", len(_documents), dim)
    except ImportError:
        logger.warning("faiss-cpu not installed — using numpy cosine similarity fallback.")
        _embeddings = vecs
        _use_faiss = False

    with open(_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(_documents, f, ensure_ascii=False, indent=2)


def _ensure_loaded():
    global _index, _documents, _embeddings, _use_faiss

    if _documents:
        return

    if _DOCS_PATH.exists():
        with open(_DOCS_PATH, encoding="utf-8") as f:
            _documents = json.load(f)

        try:
            import faiss
            if _INDEX_PATH.exists():
                _index = faiss.read_index(str(_INDEX_PATH))
                _use_faiss = True
                return
        except ImportError:
            pass

    # Rebuild from scratch if anything is missing
    build_index()


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Return the top_k most relevant documents for the given query."""
    _ensure_loaded()

    q_vec = emb.embed_query(query)   # (1, D)

    if _use_faiss and _index is not None:
        _, indices = _index.search(q_vec.astype(np.float32), top_k)
        hits = indices[0]
    else:
        scores = _cosine_sim(q_vec, _embeddings)
        hits = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in hits:
        if 0 <= idx < len(_documents):
            results.append(_documents[idx])
    return results
