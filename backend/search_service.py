"""
MemeFinder — Hybrid search service (Phase 3).

Fuses up to three retrievers into a single ranked list:

  1. SQLite FTS5 keyword search over OCR text (exact / prefix matches).
  2. BGE-M3 dense semantic search over OCR + VLM caption + humor + tags.
  3. CLIP search — text→image (multilingual) and/or image→image when an image
     is uploaded.

Results are merged with Reciprocal Rank Fusion (RRF). RRF is robust when the
underlying scores live in different scales (BM25 vs cosine vs CLIP) and only
needs the rank to combine signals.
"""
from __future__ import annotations

import asyncio
import io
import logging
import time
from collections import defaultdict
from typing import Any

from PIL import Image

from config import (
    DEFAULT_SEARCH_LIMIT,
    IMAGE_RETRIEVER_TOP_K,
    RETRIEVER_TOP_K,
    RRF_K1,
    RRF_K2,
    SEARCH_USE_FTS,
    SEARCH_USE_IMAGE_VECTOR,
    SEARCH_USE_TEXT_VECTOR,
)
import chroma_store
import clip_engine
import database as db
import text_embedder

logger = logging.getLogger(__name__)


# ── RRF ──────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    results_lists: list[list[dict]],
    k1: float = RRF_K1,
    k2: float = RRF_K2,
    weights: list[float] | None = None,
) -> list[dict]:
    """
    Merge multiple ranked candidate lists into one ranked list using RRF.

    Score per filename = sum_l(weight_l / (k1 + rank_l * k2)) where rank_l is
    the 1-based position in retriever l. The default constants (k1=60, k2=20)
    follow the TЗ — k1 dampens top-rank dominance, k2 controls rank decay.

    Parameters
    ----------
    results_lists : list of ranked candidate lists. Each item must have
        a 'filename' key. Non-list / empty entries are silently skipped.
    weights : optional per-retriever weight (defaults to 1.0 each).
    """
    score: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)

    if weights is None:
        weights = [1.0] * len(results_lists)

    for retriever_idx, rank_list in enumerate(results_lists):
        if not rank_list:
            continue
        w = weights[retriever_idx] if retriever_idx < len(weights) else 1.0
        label = f"r{retriever_idx}"
        for rank, item in enumerate(rank_list, start=1):
            filename = item.get("filename")
            if not filename:
                continue
            score[filename] += w / (k1 + rank * k2)
            sources[filename].add(label)

    fused = [
        {
            "filename": fn,
            "rrf_score": s,
            "sources": sorted(sources[fn]),
        }
        for fn, s in score.items()
    ]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


# ── Individual retrievers ────────────────────────────────────────────────────

def _fts_retriever(query: str, top_k: int) -> list[dict]:
    rows = db.search_by_text(query, limit=top_k, offset=0)
    return [{"filename": r["filename"]} for r in rows]


def _text_vector_retriever(query: str, top_k: int) -> list[dict]:
    query_emb = text_embedder.encode_one(query)
    if not query_emb:
        return []
    return chroma_store.text_search(query_emb, top_k=top_k)


def _image_text_retriever(query: str, top_k: int) -> list[dict]:
    """CLIP text→image: encode the query with CLIP-multilingual and search images."""
    query_emb = clip_engine.encode_text(query)
    if not query_emb:
        return []
    return chroma_store.image_search(query_emb, n_results=top_k)


def _image_image_retriever(image_bytes: bytes, top_k: int) -> list[dict]:
    """CLIP image→image: encode the uploaded image and search the image collection."""
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.warning("Cannot open uploaded image: %s", e)
        return []
    emb = clip_engine.encode_image(pil)
    if not emb:
        return []
    return chroma_store.image_search(emb, n_results=top_k)


# ── Public API ───────────────────────────────────────────────────────────────

async def hybrid_search(
    query: str | None = None,
    image_bytes: bytes | None = None,
    mode: str = "all",
    top_k: int = DEFAULT_SEARCH_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Run hybrid search and return a fused, paginated, metadata-enriched result.

    `mode` selects retrievers:
      - "all"   : FTS + BGE + CLIP (image if uploaded, else text→image)
      - "text"  : FTS + BGE only (faster, no CLIP — useful when image search is noisy)
      - "image" : CLIP only — needs `image_bytes` (or falls back to text→image if missing)

    The function timings every stage and exposes them in the response under
    `timing_ms`, so we can later tune k1/k2/top_k from real numbers.
    """
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()

    candidate_lists: list[list[dict]] = []
    weights: list[float] = []
    retrievers_used: list[str] = []

    has_query = bool(query and query.strip())
    has_image = bool(image_bytes)
    mode = (mode or "all").lower()
    if mode not in {"all", "text", "image"}:
        mode = "all"

    # 1) Keyword + dense text retrievers
    if has_query and mode in ("all", "text"):
        if SEARCH_USE_FTS:
            t = time.perf_counter()
            fts = await loop.run_in_executor(
                None, _fts_retriever, query.strip(), RETRIEVER_TOP_K,
            )
            timings["fts_ms"] = (time.perf_counter() - t) * 1000
            candidate_lists.append(fts)
            weights.append(1.0)
            retrievers_used.append("fts")

        if SEARCH_USE_TEXT_VECTOR:
            t = time.perf_counter()
            vector = await loop.run_in_executor(
                None, _text_vector_retriever, query.strip(), RETRIEVER_TOP_K,
            )
            timings["text_vector_ms"] = (time.perf_counter() - t) * 1000
            candidate_lists.append(vector)
            # BGE is the most powerful single signal — give it a slight bump.
            weights.append(1.2)
            retrievers_used.append("text_vector")

    # 2) Image retriever
    use_image = SEARCH_USE_IMAGE_VECTOR and mode in ("all", "image")
    if use_image:
        if has_image:
            t = time.perf_counter()
            img_results = await loop.run_in_executor(
                None, _image_image_retriever, image_bytes, IMAGE_RETRIEVER_TOP_K,
            )
            timings["image_search_ms"] = (time.perf_counter() - t) * 1000
            candidate_lists.append(img_results)
            weights.append(1.0)
            retrievers_used.append("image_image")
        elif has_query:
            t = time.perf_counter()
            img_results = await loop.run_in_executor(
                None, _image_text_retriever, query.strip(), IMAGE_RETRIEVER_TOP_K,
            )
            timings["image_text_ms"] = (time.perf_counter() - t) * 1000
            candidate_lists.append(img_results)
            # CLIP-text on Russian queries is weaker than BGE; weight it lower.
            weights.append(0.7)
            retrievers_used.append("image_text")

    if not candidate_lists:
        return {
            "results": [],
            "mode": mode,
            "retrievers": retrievers_used,
            "timing_ms": timings,
            "total_ms": (time.perf_counter() - t0) * 1000,
            "fused_total": 0,
        }

    # 3) Fuse with RRF
    t = time.perf_counter()
    fused = reciprocal_rank_fusion(candidate_lists, weights=weights)
    timings["rrf_ms"] = (time.perf_counter() - t) * 1000
    fused_total = len(fused)

    # 4) Paginate, then enrich with SQLite metadata in one batch
    page = fused[offset : offset + top_k]
    filenames = [r["filename"] for r in page]
    t = time.perf_counter()
    rows = db.get_memes_by_filenames(filenames)
    timings["metadata_ms"] = (time.perf_counter() - t) * 1000

    results = []
    for r in page:
        meme = rows.get(r["filename"])
        if not meme:
            # Drift — filename was in Chroma but missing from SQLite. Skip.
            continue
        results.append({
            "id": meme["id"],
            "filename": meme["filename"],
            "ocr_text": meme["ocr_text"],
            "thumbnail_base64": meme["thumbnail_base64"],
            "caption": meme.get("caption"),
            "humor_explain": meme.get("humor_explain"),
            "tags": meme.get("tags"),
            "score": round(r["rrf_score"], 6),
            "sources": r.get("sources", []),
        })

    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    logger.info(
        "Hybrid search: q=%r mode=%s retrievers=%s fused=%d returned=%d total=%.1fms",
        (query or "")[:40], mode, retrievers_used, fused_total, len(results),
        timings["total_ms"],
    )

    return {
        "results": results,
        "mode": mode,
        "retrievers": retrievers_used,
        "timing_ms": {k: round(v, 2) for k, v in timings.items()},
        "fused_total": fused_total,
    }
