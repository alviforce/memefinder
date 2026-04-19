"""
MemeFinder — FastAPI application
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import DEFAULT_SEARCH_LIMIT, MEMES_DIR
import database as db
import clip_engine
import ocr_engine
import chroma_store
import indexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    logger.info("Database ready. Memes: %d, embeddings: %d",
                db.get_meme_count(), chroma_store.get_count())

    logger.info("Loading ML models...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, clip_engine.load_models)
    await loop.run_in_executor(None, ocr_engine.load_model)
    logger.info("ML models loaded.")

    yield

    logger.info("Shutdown complete.")


app = FastAPI(
    title="MemeFinder API",
    description="Search memes by text (OCR) or meaning (CLIP)",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/search")
def search_memes(
    q: str = Query(..., min_length=1),
    mode: str = Query("clip", pattern="^(ocr|clip)$"),
    limit: int = Query(DEFAULT_SEARCH_LIMIT, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
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
                    "score": None,
                }
                for r in rows
            ],
        }

    # CLIP search
    query_embedding = clip_engine.encode_text(q)
    if not query_embedding:
        raise HTTPException(500, "Failed to encode query")

    matches = chroma_store.search(query_embedding, n_results=limit)
    if not matches:
        return {"mode": "clip", "query": q, "results": []}

    # Single batch query instead of N+1
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
                "score": round(1 - distance_by_filename[filename], 4),
            })

    return {"mode": "clip", "query": q, "results": results}


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


@app.post("/api/index/start")
async def start_indexing():
    if indexer.indexing_state["status"] == "running":
        return {"message": "Already running", **indexer.indexing_state}
    asyncio.create_task(indexer.run_indexing())
    return {"message": "Indexing started"}


@app.post("/api/index/resync")
async def resync_stores():
    """
    Detect and fix divergence between SQLite and ChromaDB.
    - Removes ChromaDB entries whose filenames are absent from SQLite.
    - Returns a count of removed stale entries.
    """
    sqlite_filenames = db.get_indexed_filenames()
    chroma_filenames = chroma_store.get_indexed_filenames()

    stale = list(chroma_filenames - sqlite_filenames)
    if stale:
        chroma_store.delete_embeddings(stale)
        logger.info("Resync: removed %d stale ChromaDB entries", len(stale))

    missing_embeddings = sqlite_filenames - chroma_filenames
    return {
        "stale_removed": len(stale),
        "missing_embeddings": len(missing_embeddings),
        "message": (
            f"Removed {len(stale)} stale entries. "
            f"{len(missing_embeddings)} memes have no embedding (re-index to fix)."
        ),
    }


@app.get("/api/stats")
def get_stats():
    return {
        "meme_count": db.get_meme_count(),
        "embedding_count": chroma_store.get_count(),
        "indexing": indexer.indexing_state,
    }
