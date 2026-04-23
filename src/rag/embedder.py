"""
Sentence-transformer embedder with lazy loading and numpy fallback.
Falls back to TF-IDF cosine similarity if sentence-transformers is unavailable.
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"   # 25 MB, runs locally, no API key
_embedder = None
_use_tfidf = False


def _load():
    global _embedder, _use_tfidf
    if _embedder is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(_MODEL_NAME)
        logger.info("sentence-transformers loaded: %s", _MODEL_NAME)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — falling back to TF-IDF embeddings. "
            "Run: pip install sentence-transformers"
        )
        _use_tfidf = True
        _embedder = _TFIDFEmbedder()


class _TFIDFEmbedder:
    """Minimal TF-IDF fallback so RAG works without sentence-transformers."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(max_features=512, stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]):
        self._vec.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str] | str, **_) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not self._fitted:
            self._vec.fit(texts)
            self._fitted = True
        return self._vec.transform(texts).toarray().astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return (N, D) float32 array of embeddings for a list of strings."""
    _load()
    if _use_tfidf:
        return _embedder.encode(texts)
    return _embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def embed_query(text: str) -> np.ndarray:
    """Return (1, D) float32 array for a single query string."""
    _load()
    if _use_tfidf:
        return _embedder.encode([text])
    result = _embedder.encode([text], convert_to_numpy=True, show_progress_bar=False)
    return result


def fit_tfidf(texts: list[str]):
    """Pre-fit the TF-IDF vocabulary if using the fallback embedder."""
    _load()
    if _use_tfidf:
        _embedder.fit(texts)
