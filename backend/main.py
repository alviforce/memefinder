"""
MemeFinder — FastAPI application
"""
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx

from config import DEFAULT_SEARCH_LIMIT, MEMES_DIR
import database as db
import clip_engine
import ocr_engine
import vlm_engine
import chroma_store
import indexer
import search_service
import text_embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    logger.info(
        "Database ready. Memes: %d, image embeddings: %d, text embeddings: %d",
        db.get_meme_count(),
        chroma_store.get_image_count(),
        chroma_store.get_text_count(),
    )

    logger.info("Loading ML models...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, clip_engine.load_models)
    await loop.run_in_executor(None, ocr_engine.load_model)
    await loop.run_in_executor(None, vlm_engine.load_model)
    await loop.run_in_executor(None, text_embedder.load_model)
    logger.info("ML models loaded.")

    yield

    logger.info("Shutdown complete.")


app = FastAPI(
    title="MemeFinder API",
    description="Hybrid meme search: FTS5 + BGE-M3 + CLIP (Phase 3)",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_tags(raw: str | None) -> list[str]:
    """Tags are stored as JSON list (or comma-separated); normalize to list."""
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        try:
            value = json.loads(raw)
            if isinstance(value, list):
                return [str(t).strip() for t in value if str(t).strip()]
        except json.JSONDecodeError:
            pass
    return [t.strip() for t in raw.split(",") if t.strip()]


def _enrich_search_payload(payload: dict, query: str | None) -> dict:
    """Convert DB tag-strings to lists in the response."""
    for r in payload.get("results", []):
        r["tags"] = _parse_tags(r.get("tags"))
    payload["query"] = query
    return payload


# ── Search ───────────────────────────────────────────────────────────────────

@app.get("/api/search")
def search_memes_legacy(
    q: str = Query(..., min_length=1),
    mode: str = Query("clip", pattern="^(ocr|clip)$"),
    limit: int = Query(DEFAULT_SEARCH_LIMIT, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    Legacy single-retriever search kept for backward compatibility.
    Use POST /api/search for hybrid (FTS+BGE+CLIP) and image queries.
    """
    if mode == "ocr":
        rows = db.search_by_text(q, limit=limit, offset=offset)
        return {
            "mode": "ocr",
            "query": q,
            "results": [
                {
                    "id": r["id"],
                    "filename": r["filename"],
                    "ocr_text": r["ocr_text"],
                    "thumbnail_base64": r["thumbnail_base64"],
                    "caption": r.get("caption"),
                    "humor_explain": r.get("humor_explain"),
                    "tags": _parse_tags(r.get("tags")),
                    "score": None,
                }
                for r in rows
            ],
        }

    # CLIP search
    query_embedding = clip_engine.encode_text(q)
    if not query_embedding:
        raise HTTPException(500, "Failed to encode query")

    matches = chroma_store.image_search(query_embedding, n_results=limit)
    if not matches:
        return {"mode": "clip", "query": q, "results": []}

    filenames = [m["filename"] for m in matches]
    memes_by_filename = db.get_memes_by_filenames(filenames)
    distance_by_filename = {m["filename"]: m["distance"] for m in matches}

    results = []
    for filename in filenames:
        meme = memes_by_filename.get(filename)
        if meme:
            results.append({
                "id": meme["id"],
                "filename": meme["filename"],
                "ocr_text": meme["ocr_text"],
                "thumbnail_base64": meme["thumbnail_base64"],
                "caption": meme.get("caption"),
                "humor_explain": meme.get("humor_explain"),
                "tags": _parse_tags(meme.get("tags")),
                "score": round(1 - distance_by_filename[filename], 4),
            })

    return {"mode": "clip", "query": q, "results": results}


@app.post("/api/search")
async def search_memes_hybrid(
    query: str | None = Form(None),
    mode: str = Form("all"),
    limit: int = Form(DEFAULT_SEARCH_LIMIT, ge=1, le=100),
    offset: int = Form(0, ge=0),
    image: UploadFile | None = File(None),
):
    """
    Hybrid search: combines FTS5 + BGE-M3 + CLIP via Reciprocal Rank Fusion.
    Accepts text query and/or an image upload (multipart/form-data).
    """
    image_bytes = await image.read() if image else None

    if not (query and query.strip()) and not image_bytes:
        raise HTTPException(400, "Either 'query' or 'image' is required")

    payload = await search_service.hybrid_search(
        query=query,
        image_bytes=image_bytes,
        mode=mode,
        top_k=limit,
        offset=offset,
    )
    return _enrich_search_payload(payload, query)


# ── Static images ────────────────────────────────────────────────────────────

@app.get("/api/image/original/{filename:path}")
async def get_original_image(filename: str):
    # Prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")

    file_path = MEMES_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "Image not found")

    return FileResponse(
        path=file_path,
        headers={"Cache-Control": "public, max-age=31536000"},
    )


# ── Indexing & maintenance ───────────────────────────────────────────────────

@app.post("/api/index/start")
async def start_indexing():
    if indexer.indexing_state["status"] == "running":
        return {"message": "Already running", **indexer.indexing_state}
    asyncio.create_task(indexer.run_indexing())
    return {"message": "Indexing started"}


@app.post("/api/index/vlm")
async def start_vlm_backfill(force: bool = False):
    """
    Run VLM captioning on every already-indexed meme that has no caption yet.
    Fills caption/humor_explain/tags and merges any newly-detected text into ocr_text.
    """
    if indexer.vlm_state["status"] == "running":
        return {"message": "VLM backfill already running", **indexer.vlm_state}
    if indexer.indexing_state["status"] == "running":
        raise HTTPException(409, "Indexing is running — wait for it to finish")
    if indexer.reocr_state["status"] == "running":
        raise HTTPException(409, "Re-OCR is running — wait for it to finish")
    try:
        # Стучимся напрямую в Ollama, как мы делали через терминал
        with httpx.Client() as client:
            resp = client.get("http://localhost:11434/api/tags", timeout=5.0)
            
        if resp.status_code != 200:
            raise HTTPException(503, "Ollama server is not responding correctly")
            
        models_data = resp.json().get("models", [])
        available_models = [m["name"] for m in models_data]
        
        target_model = "huihui_ai/qwen3-vl-abliterated:8b-instruct"
        
        # Проверяем, есть ли наша модель в списке скачанных
        if target_model not in available_models:
             raise HTTPException(
                 503, 
                 f"Model {target_model} not found in Ollama. Please run 'ollama pull {target_model}'"
             )
             
    except httpx.RequestError as e:
        raise HTTPException(
            503, 
            f"Cannot connect to Ollama at http://localhost:11434. Error: {e}"
        )
    asyncio.create_task(indexer.run_vlm_backfill(force=force))
    return {"message": "VLM backfill started"}


@app.post("/api/index/vlm/stragglers")
async def start_vlm_stragglers():
    """
    Last-resort pass for memes still without a caption after the regular VLM
    backfill. Uses a compact prompt (text + caption only) with num_predict=3072
    so dense study screenshots and stylized fonts get a chance.

    On final failure the meme is stamped with a sentinel caption so future
    runs don't re-pick it. Safe to run multiple times — it only touches rows
    where caption IS NULL or empty.
    """
    if indexer.straggler_state["status"] == "running":
        return {
            "message": "Straggler backfill already running",
            **indexer.straggler_state,
        }
    if indexer.indexing_state["status"] == "running":
        raise HTTPException(409, "Indexing is running — wait for it to finish")
    if indexer.vlm_state["status"] == "running":
        raise HTTPException(409, "VLM backfill is running — wait for it to finish")
    if indexer.reocr_state["status"] == "running":
        raise HTTPException(409, "Re-OCR is running — wait for it to finish")

    # Reuse the same Ollama health-check as /api/index/vlm so we fail fast
    # if Ollama isn't reachable or the model is missing.
    try:
        with httpx.Client() as client:
            resp = client.get("http://localhost:11434/api/tags", timeout=5.0)
        if resp.status_code != 200:
            raise HTTPException(503, "Ollama server is not responding correctly")
        models_data = resp.json().get("models", [])
        available_models = [m["name"] for m in models_data]
        target_model = "huihui_ai/qwen3-vl-abliterated:8b-instruct"
        if target_model not in available_models:
            raise HTTPException(
                503,
                f"Model {target_model} not found in Ollama. "
                f"Run 'ollama pull {target_model}'",
            )
    except httpx.RequestError as e:
        raise HTTPException(
            503,
            f"Cannot connect to Ollama at http://localhost:11434. Error: {e}",
        )

    asyncio.create_task(indexer.run_vlm_stragglers())
    return {"message": "Straggler backfill started"}


@app.post("/api/index/reocr")
async def start_reocr():
    """
    Re-run OCR on all already-indexed memes. Use after changing OCR_ENGINE
    or PREPROCESS to refresh stored OCR text. Embeddings/thumbnails are kept.
    """
    if indexer.reocr_state["status"] == "running":
        return {"message": "Re-OCR already running", **indexer.reocr_state}
    if indexer.indexing_state["status"] == "running":
        raise HTTPException(409, "Indexing is running — wait for it to finish")
    asyncio.create_task(indexer.run_reocr())
    return {"message": "Re-OCR started"}


@app.post("/api/index/text-embeddings")
async def start_text_embed_backfill():
    """
    Build / refresh BGE-M3 dense text embeddings for every indexed meme.
    Use after enabling Phase 3 hybrid search on an existing database.
    """
    if indexer.text_embed_state["status"] == "running":
        return {
            "message": "Text-embed backfill already running",
            **indexer.text_embed_state,
        }
    if indexer.indexing_state["status"] == "running":
        raise HTTPException(409, "Indexing is running — wait for it to finish")
    asyncio.create_task(indexer.run_text_embed_backfill())
    return {"message": "Text-embed backfill started"}


@app.post("/api/index/resync")
async def resync_stores():
    """
    Detect and fix divergence between SQLite and ChromaDB.
    - Removes stale entries from both image and text Chroma collections whose
      filenames are absent from SQLite.
    - Returns counts so the caller knows which collections are still short.
    """
    sqlite_filenames = db.get_indexed_filenames()
    chroma_image = chroma_store.get_image_filenames()
    chroma_text = chroma_store.get_text_filenames()

    stale_image = list(chroma_image - sqlite_filenames)
    if stale_image:
        chroma_store.delete_image_embeddings(stale_image)
        logger.info("Resync: removed %d stale image embeddings", len(stale_image))

    stale_text = list(chroma_text - sqlite_filenames)
    if stale_text:
        chroma_store.delete_text_embeddings(stale_text)
        logger.info("Resync: removed %d stale text embeddings", len(stale_text))

    missing_image = sqlite_filenames - chroma_image
    missing_text = sqlite_filenames - chroma_text
    return {
        "stale_image_removed": len(stale_image),
        "stale_text_removed": len(stale_text),
        "missing_image_embeddings": len(missing_image),
        "missing_text_embeddings": len(missing_text),
        # Backwards-compat alias for the old field name
        "stale_removed": len(stale_image),
        "missing_embeddings": len(missing_image),
        "message": (
            f"Removed {len(stale_image)} stale image, {len(stale_text)} stale text. "
            f"{len(missing_image)} memes missing image embeddings, "
            f"{len(missing_text)} memes missing text embeddings."
        ),
    }


@app.get("/api/stats")
def get_stats():
    return {
        "meme_count": db.get_meme_count(),
        "embedding_count": chroma_store.get_image_count(),
        "text_embedding_count": chroma_store.get_text_count(),
        "indexing": indexer.indexing_state,
        "reocr": indexer.reocr_state,
        "vlm": indexer.vlm_state,
        "vlm_stragglers": indexer.straggler_state,
        "text_embed": indexer.text_embed_state,
    }
