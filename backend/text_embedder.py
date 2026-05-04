"""
MemeFinder — Text embedder (BGE-M3, multilingual).

Encodes the canonical meme text (`ocr + caption + humor + tags`) into a 1024-d
dense vector that lives in ChromaDB collection `text_embeddings`. Used for
semantic search alongside FTS5 keyword and CLIP image search.

The model is multilingual (ru/en/100+), normalized so cosine similarity is the
ChromaDB default (`hnsw:space=cosine`).
"""
from __future__ import annotations

import gc
import logging

from config import BATCH_SIZE, DEVICE, TEXT_EMBED_MODEL_NAME

logger = logging.getLogger(__name__)

_model = None


def load_model() -> None:
    """Load BGE-M3 once at startup."""
    global _model
    if _model is not None:
        return
    from sentence_transformers import SentenceTransformer
    logger.info("Loading text embedder: %s on %s", TEXT_EMBED_MODEL_NAME, DEVICE)
    _model = SentenceTransformer(TEXT_EMBED_MODEL_NAME, device=DEVICE)
    logger.info("Text embedder ready (dim=%d).", _model.get_sentence_embedding_dimension())


def _get_model():
    if _model is None:
        load_model()
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    """
    Encode a batch of strings to normalized dense vectors.
    Empty inputs are filtered out before encoding; returned list lines up 1:1
    with `texts` (zero-vector substitute for empty/blank inputs would skew
    cosine similarity, so we instead drop them at the call site).
    """
    if not texts:
        return []
    try:
        model = _get_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )
        return embeddings.tolist()
    except Exception as e:
        logger.error("Text encoding failed: %s", e)
        return [[] for _ in texts]
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


def encode_one(text: str) -> list[float]:
    """Encode a single string to a normalized dense vector."""
    if not text or not text.strip():
        return []
    out = encode([text])
    return out[0] if out else []
