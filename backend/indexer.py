"""
MemeFinder — Local Folder Indexer
Reads photos from the 'memes' folder, generates thumbnails,
runs OCR + CLIP in batches, stores results in SQLite + ChromaDB.

_process_batch runs in a thread pool so it never blocks the async event loop.
"""
import asyncio
import base64
import gc
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from config import BATCH_SIZE, INDEX_LIMIT, THUMBNAIL_SIZE, THUMBNAIL_QUALITY, MEMES_DIR
import database as db
import clip_engine
import ocr_engine
import chroma_store

logger = logging.getLogger(__name__)

# Single thread worker — ML models are not thread-safe (GPU state)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexer")

indexing_state = {
    "status": "idle",   # idle | running | done | error
    "total": 0,
    "processed": 0,
    "errors": 0,
    "started_at": None,
    "elapsed_seconds": 0,
}

reocr_state = {
    "status": "idle",   # idle | running | done | error
    "total": 0,
    "processed": 0,
    "updated": 0,
    "errors": 0,
    "started_at": None,
    "elapsed_seconds": 0,
}


def _generate_thumbnail_base64(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=THUMBNAIL_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _embed_only_batch(batch_items: list[dict]) -> tuple[int, int]:
    """
    Fast path for files already in SQLite but missing ChromaDB embeddings.
    Skips OCR and thumbnail — only encodes images and upserts to ChromaDB.
    Returns (saved_count, error_count).
    """
    if not batch_items:
        return 0, 0

    pil_images = []
    for item in batch_items:
        try:
            pil_images.append(Image.open(io.BytesIO(item["image_bytes"])).convert("RGB"))
        except Exception as e:
            logger.warning("Cannot open %s: %s", item["filename"], e)
            pil_images.append(Image.new("RGB", (224, 224)))

    embeddings = clip_engine.encode_images_batch(pil_images)

    chroma_ids, chroma_embeddings, chroma_metadatas = [], [], []
    errors = 0
    for i, item in enumerate(batch_items):
        embedding = embeddings[i] if i < len(embeddings) else []
        if embedding:
            chroma_ids.append(item["filename"])
            chroma_embeddings.append(embedding)
            chroma_metadatas.append({"filename": item["filename"]})
        else:
            logger.warning("No embedding for %s", item["filename"])
            errors += 1

    if chroma_ids:
        try:
            chroma_store.add_embeddings_batch(chroma_ids, chroma_embeddings, chroma_metadatas)
        except Exception as e:
            logger.error("ChromaDB upsert failed: %s", e)
            errors += len(chroma_ids)

    del pil_images
    gc.collect()

    saved = len(batch_items) - errors
    logger.info("Embed-only batch done: %d embedded, %d errors", saved, errors)
    return saved, errors


def _process_batch(batch_items: list[dict]) -> tuple[int, int]:
    """
    Synchronous heavy work: CLIP encode + OCR + persist.
    Returns (saved_count, error_count).
    Runs in a thread pool — never call directly from async code.
    """
    if not batch_items:
        return 0, 0

    pil_images = []
    for item in batch_items:
        try:
            pil_images.append(Image.open(io.BytesIO(item["image_bytes"])).convert("RGB"))
        except Exception as e:
            logger.warning("Cannot open %s: %s", item["filename"], e)
            pil_images.append(Image.new("RGB", (224, 224)))

    embeddings = clip_engine.encode_images_batch(pil_images)
    ocr_texts = ocr_engine.extract_texts_batch([item["image_bytes"] for item in batch_items])

    saved = 0
    errors = 0
    chroma_ids, chroma_embeddings, chroma_metadatas = [], [], []

    for i, item in enumerate(batch_items):
        ocr_text = ocr_texts[i] if i < len(ocr_texts) else ""
        embedding = embeddings[i] if i < len(embeddings) else []

        try:
            thumbnail_b64 = _generate_thumbnail_base64(item["image_bytes"])
        except Exception as e:
            logger.warning("Thumbnail failed for %s: %s", item["filename"], e)
            errors += 1
            continue

        meme_id = db.insert_meme(
            filename=item["filename"],
            ocr_text=ocr_text,
            thumbnail_base64=thumbnail_b64,
        )

        if meme_id and embedding:
            chroma_ids.append(item["filename"])
            chroma_embeddings.append(embedding)
            chroma_metadatas.append({"filename": item["filename"]})
        elif meme_id and not embedding:
            logger.warning("No embedding for %s — will be missing from CLIP search", item["filename"])

        saved += 1

    if chroma_ids:
        try:
            chroma_store.add_embeddings_batch(chroma_ids, chroma_embeddings, chroma_metadatas)
        except Exception as e:
            logger.error("ChromaDB upsert failed for batch: %s", e)
            errors += len(chroma_ids)

    del pil_images
    gc.collect()

    logger.info("Batch done: %d saved, %d errors", saved, errors)
    return saved, errors


def _reocr_batch(batch_items: list[dict]) -> tuple[int, int]:
    """
    Re-run OCR for files already present in SQLite and update their ocr_text.
    Does NOT touch ChromaDB or thumbnails — embeddings don't depend on OCR.
    Returns (updated_count, error_count).
    """
    if not batch_items:
        return 0, 0

    try:
        ocr_texts = ocr_engine.extract_texts_batch(
            [item["image_bytes"] for item in batch_items]
        )
    except Exception as e:
        logger.error("Re-OCR batch failed: %s", e)
        return 0, len(batch_items)

    updated = 0
    errors = 0
    for i, item in enumerate(batch_items):
        ocr_text = ocr_texts[i] if i < len(ocr_texts) else ""
        try:
            if db.update_ocr_text(item["filename"], ocr_text):
                updated += 1
            else:
                errors += 1
        except Exception as e:
            logger.warning("DB update failed for %s: %s", item["filename"], e)
            errors += 1

    gc.collect()
    logger.info("Re-OCR batch done: %d updated, %d errors", updated, errors)
    return updated, errors


async def run_reocr():
    """
    Re-run OCR on every already-indexed meme whose file still exists on disk.
    Useful after changing OCR_ENGINE / PREPROCESS to refresh stored OCR text
    without redoing embeddings or thumbnails.
    """
    global reocr_state

    if reocr_state["status"] == "running":
        logger.warning("Re-OCR already running.")
        return
    if indexing_state["status"] == "running":
        logger.warning("Cannot start re-OCR while indexing is running.")
        return

    reocr_state.update({
        "status": "running",
        "total": 0,
        "processed": 0,
        "updated": 0,
        "errors": 0,
        "started_at": time.time(),
        "elapsed_seconds": 0,
    })

    loop = asyncio.get_event_loop()

    try:
        sqlite_filenames = db.get_indexed_filenames()
        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
        files_on_disk = {
            e.name: e for e in MEMES_DIR.iterdir()
            if e.is_file() and e.suffix.lower() in valid_ext
        } if MEMES_DIR.exists() else {}

        # Re-OCR only files present in BOTH SQLite and on disk.
        targets = [files_on_disk[name] for name in sqlite_filenames if name in files_on_disk]
        reocr_state["total"] = len(targets)
        logger.info("Re-OCR: %d files to reprocess", len(targets))

        batch: list[dict] = []
        processed_count = 0

        async def _flush():
            nonlocal processed_count
            if not batch:
                return
            updated, errors = await loop.run_in_executor(_executor, _reocr_batch, list(batch))
            processed_count += len(batch)
            reocr_state["processed"] = processed_count
            reocr_state["updated"] += updated
            reocr_state["errors"] += errors
            batch.clear()
            reocr_state["elapsed_seconds"] = time.time() - reocr_state["started_at"]

        for file_path in targets:
            try:
                image_bytes = file_path.read_bytes()
            except Exception as e:
                logger.error("Cannot read %s: %s", file_path.name, e)
                reocr_state["errors"] += 1
                processed_count += 1
                reocr_state["processed"] = processed_count
                continue

            batch.append({"filename": file_path.name, "image_bytes": image_bytes})
            if len(batch) >= BATCH_SIZE:
                await _flush()

        await _flush()

        reocr_state["status"] = "done"
        reocr_state["elapsed_seconds"] = time.time() - reocr_state["started_at"]
        logger.info(
            "Re-OCR complete: %d processed, %d updated, %d errors, %.1fs",
            reocr_state["processed"],
            reocr_state["updated"],
            reocr_state["errors"],
            reocr_state["elapsed_seconds"],
        )

    except Exception as e:
        logger.error("Re-OCR failed: %s", e, exc_info=True)
        reocr_state["status"] = "error"
        reocr_state["elapsed_seconds"] = time.time() - reocr_state["started_at"]


async def run_indexing():
    """Async indexing pipeline. CPU work runs in thread pool."""
    global indexing_state

    if indexing_state["status"] == "running":
        logger.warning("Indexing already running.")
        return

    indexing_state.update({
        "status": "running",
        "total": 0,
        "processed": 0,
        "errors": 0,
        "started_at": time.time(),
        "elapsed_seconds": 0,
    })

    loop = asyncio.get_event_loop()

    try:
        sqlite_filenames = db.get_indexed_filenames()
        chroma_filenames = chroma_store.get_indexed_filenames()
        logger.info(
            "Already indexed: %d in SQLite, %d in ChromaDB",
            len(sqlite_filenames), len(chroma_filenames),
        )

        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
        files_to_process = [
            e for e in MEMES_DIR.iterdir()
            if e.is_file() and e.suffix.lower() in valid_ext
        ] if MEMES_DIR.exists() else []

        indexing_state["total"] = len(files_to_process)
        logger.info("Found %d image files to process", len(files_to_process))

        # Three queues:
        #   full_batch  — new files: need OCR + thumbnail + embedding
        #   embed_batch — in SQLite but missing ChromaDB embedding (embed-only)
        full_batch: list[dict] = []
        embed_batch: list[dict] = []
        processed_count = 0

        async def _flush_full():
            nonlocal processed_count
            if not full_batch:
                return
            saved, errors = await loop.run_in_executor(_executor, _process_batch, list(full_batch))
            processed_count += len(full_batch)
            indexing_state["processed"] = processed_count
            indexing_state["errors"] += errors
            full_batch.clear()
            indexing_state["elapsed_seconds"] = time.time() - indexing_state["started_at"]

        async def _flush_embed():
            nonlocal processed_count
            if not embed_batch:
                return
            saved, errors = await loop.run_in_executor(_executor, _embed_only_batch, list(embed_batch))
            processed_count += len(embed_batch)
            indexing_state["processed"] = processed_count
            indexing_state["errors"] += errors
            embed_batch.clear()
            indexing_state["elapsed_seconds"] = time.time() - indexing_state["started_at"]

        for file_path in files_to_process:
            # INDEX_LIMIT = 0 means no limit
            if INDEX_LIMIT > 0 and processed_count >= INDEX_LIMIT:
                logger.info("Reached INDEX_LIMIT=%d", INDEX_LIMIT)
                break

            filename = file_path.name

            in_sqlite = filename in sqlite_filenames
            in_chroma = filename in chroma_filenames

            if in_sqlite and in_chroma:
                # Fully indexed — skip
                processed_count += 1
                indexing_state["processed"] = processed_count
                continue

            try:
                image_bytes = file_path.read_bytes()
            except Exception as e:
                logger.error("Cannot read %s: %s", filename, e)
                indexing_state["errors"] += 1
                continue

            if in_sqlite and not in_chroma:
                # Has OCR/thumbnail but missing embedding — fast path
                embed_batch.append({"filename": filename, "image_bytes": image_bytes})
                if len(embed_batch) >= BATCH_SIZE:
                    await _flush_embed()
            else:
                # Completely new file — full processing
                full_batch.append({"filename": filename, "image_bytes": image_bytes})
                if len(full_batch) >= BATCH_SIZE:
                    await _flush_full()

        # Flush remaining
        await _flush_full()
        await _flush_embed()

        indexing_state["status"] = "done"
        indexing_state["elapsed_seconds"] = time.time() - indexing_state["started_at"]
        logger.info(
            "Indexing complete: %d processed, %d errors, %.1fs",
            indexing_state["processed"],
            indexing_state["errors"],
            indexing_state["elapsed_seconds"],
        )

    except Exception as e:
        logger.error("Indexing failed: %s", e, exc_info=True)
        indexing_state["status"] = "error"
        indexing_state["elapsed_seconds"] = time.time() - indexing_state["started_at"]
