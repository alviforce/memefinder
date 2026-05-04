"""
MemeFinder — Local Folder Indexer
Reads photos from the 'memes' folder, generates thumbnails,
runs OCR + VLM + CLIP + BGE-M3 in batches, stores results in SQLite + ChromaDB.

_process_batch runs in a thread pool so it never blocks the async event loop.
"""
import asyncio
import base64
import gc
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from config import BATCH_SIZE, INDEX_LIMIT, THUMBNAIL_SIZE, THUMBNAIL_QUALITY, MEMES_DIR
import database as db
import clip_engine
import ocr_engine
import chroma_store
import text_embedder
import vlm_engine

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

vlm_state = {
    "status": "idle",   # idle | running | done | error
    "total": 0,
    "processed": 0,
    "updated": 0,
    "errors": 0,
    "started_at": None,
    "elapsed_seconds": 0,
}

text_embed_state = {
    "status": "idle",   # idle | running | done | error
    "total": 0,
    "processed": 0,
    "updated": 0,
    "errors": 0,
    "started_at": None,
    "elapsed_seconds": 0,
}

straggler_state = {
    "status": "idle",   # idle | running | done | error
    "total": 0,
    "processed": 0,
    "updated": 0,        # успешно описаны компактным промптом
    "marked_skipped": 0, # помечены '[авто: не удалось распознать]' после фейла
    "errors": 0,
    "started_at": None,
    "elapsed_seconds": 0,
}

# Sentinel caption written when even the compact prompt fails — keeps the meme
# out of future stragglers queues but is recognizable in search results.
SKIPPED_CAPTION_SENTINEL = "[авто: не удалось распознать]"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _merge_ocr_with_vlm_text(ocr_text: str, vlm_text: str) -> str:
    """Append VLM-recognized text to OCR text, skipping if it's a substring."""
    ocr_text = (ocr_text or "").strip()
    vlm_text = (vlm_text or "").strip()
    if not vlm_text:
        return ocr_text
    if not ocr_text:
        return vlm_text
    if vlm_text.lower() in ocr_text.lower():
        return ocr_text
    return f"{ocr_text} {vlm_text}"


def _vlm_fields(
    image_bytes: bytes, filename: str | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Run VLM on a single image. Returns (caption, humor, tags_json, meme_text)
    — all None when VLM is unavailable or failed. `filename` is only used to
    enrich VLM logs with file context.
    """
    result = vlm_engine.describe_meme(image_bytes, filename=filename)
    if not result:
        return None, None, None, None
    tags_json = json.dumps(result.get("tags", []), ensure_ascii=False) if result.get("tags") else None
    return (
        result.get("caption") or None,
        result.get("humor") or None,
        tags_json,
        result.get("text") or None,
    )


def _generate_thumbnail_base64(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=THUMBNAIL_QUALITY, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _embed_texts_for_filenames(filenames: list[str], text_blobs: list[str]) -> int:
    """
    Encode `text_blobs` with BGE-M3 and upsert into ChromaDB text_embeddings.
    Returns number of vectors actually stored (skips empty blobs).
    """
    pairs = [(fn, txt) for fn, txt in zip(filenames, text_blobs) if txt and txt.strip()]
    if not pairs:
        return 0
    fns, blobs = zip(*pairs)
    embeddings = text_embedder.encode(list(blobs))
    chroma_store.add_text_embeddings_batch(
        ids=list(fns),
        embeddings=embeddings,
        metadatas=[{"filename": fn} for fn in fns],
    )
    return sum(1 for e in embeddings if e)


# ── Per-batch workers (run in thread pool) ───────────────────────────────────

def _embed_only_batch(batch_items: list[dict]) -> tuple[int, int]:
    """
    Fast path for files already in SQLite but missing ChromaDB image embeddings.
    Skips OCR and thumbnail — only encodes images and upserts to ChromaDB.
    Also (re)generates the BGE text embedding from existing SQLite metadata,
    so resync covers both vector stores in one go.
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
            chroma_store.add_image_embeddings_batch(chroma_ids, chroma_embeddings, chroma_metadatas)
        except Exception as e:
            logger.error("ChromaDB upsert failed: %s", e)
            errors += len(chroma_ids)

    # Also backfill text embeddings for these rows.
    text_blobs = [item.get("text_for_embedding", "") for item in batch_items]
    try:
        _embed_texts_for_filenames(
            [item["filename"] for item in batch_items], text_blobs,
        )
    except Exception as e:
        logger.error("Text embedding backfill failed in embed-only batch: %s", e)

    del pil_images
    gc.collect()

    saved = len(batch_items) - errors
    logger.info("Embed-only batch done: %d embedded, %d errors", saved, errors)
    return saved, errors


def _process_batch(batch_items: list[dict]) -> tuple[int, int]:
    """
    Synchronous heavy work: CLIP encode + OCR + VLM + persist + BGE text embed.
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
    text_blob_filenames: list[str] = []
    text_blobs: list[str] = []

    for i, item in enumerate(batch_items):
        ocr_text = ocr_texts[i] if i < len(ocr_texts) else ""
        embedding = embeddings[i] if i < len(embeddings) else []

        try:
            thumbnail_b64 = _generate_thumbnail_base64(item["image_bytes"])
        except Exception as e:
            logger.warning("Thumbnail failed for %s: %s", item["filename"], e)
            errors += 1
            continue

        # VLM — sequential per image (no batching support in Ollama)
        caption, humor, tags_json, vlm_text = _vlm_fields(item["image_bytes"])
        merged_text = _merge_ocr_with_vlm_text(ocr_text, vlm_text or "")
        text_for_embedding = db.build_text_for_embedding(
            merged_text, caption, humor, tags_json,
        )

        meme_id = db.insert_meme(
            filename=item["filename"],
            ocr_text=merged_text,
            thumbnail_base64=thumbnail_b64,
            caption=caption,
            humor_explain=humor,
            tags=tags_json,
            text_for_embedding=text_for_embedding,
        )

        if meme_id and embedding:
            chroma_ids.append(item["filename"])
            chroma_embeddings.append(embedding)
            chroma_metadatas.append({"filename": item["filename"]})
        elif meme_id and not embedding:
            logger.warning("No embedding for %s — will be missing from CLIP search", item["filename"])

        if meme_id and text_for_embedding:
            text_blob_filenames.append(item["filename"])
            text_blobs.append(text_for_embedding)

        saved += 1

    if chroma_ids:
        try:
            chroma_store.add_image_embeddings_batch(chroma_ids, chroma_embeddings, chroma_metadatas)
        except Exception as e:
            logger.error("ChromaDB image upsert failed for batch: %s", e)
            errors += len(chroma_ids)

    if text_blob_filenames:
        try:
            _embed_texts_for_filenames(text_blob_filenames, text_blobs)
        except Exception as e:
            logger.error("BGE text embedding upsert failed for batch: %s", e)
            # Don't bump errors — image-side persist still succeeded.

    del pil_images
    gc.collect()

    logger.info("Batch done: %d saved, %d errors", saved, errors)
    return saved, errors


def _reocr_batch(batch_items: list[dict]) -> tuple[int, int]:
    """
    Re-run OCR for files already present in SQLite and update their ocr_text.
    Refreshes text_for_embedding via db.update_ocr_text and re-embeds with BGE.
    Image embeddings/thumbnails are kept untouched.
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
    text_blob_filenames: list[str] = []
    text_blobs: list[str] = []
    for i, item in enumerate(batch_items):
        ocr_text = ocr_texts[i] if i < len(ocr_texts) else ""
        try:
            if db.update_ocr_text(item["filename"], ocr_text):
                updated += 1
                refreshed = db.get_meme_by_filename(item["filename"])
                if refreshed and refreshed.get("text_for_embedding"):
                    text_blob_filenames.append(item["filename"])
                    text_blobs.append(refreshed["text_for_embedding"])
            else:
                errors += 1
        except Exception as e:
            logger.warning("DB update failed for %s: %s", item["filename"], e)
            errors += 1

    if text_blob_filenames:
        try:
            _embed_texts_for_filenames(text_blob_filenames, text_blobs)
        except Exception as e:
            logger.error("Re-OCR BGE re-embed failed: %s", e)

    gc.collect()
    logger.info("Re-OCR batch done: %d updated, %d errors", updated, errors)
    return updated, errors


def _vlm_backfill_batch(batch_items: list[dict]) -> tuple[int, int]:
    """
    Run VLM for files already in SQLite and fill caption/humor/tags.
    Also merges VLM-detected text into ocr_text and refreshes the BGE embedding.
    """
    if not batch_items:
        return 0, 0

    updated = 0
    errors = 0
    text_blob_filenames: list[str] = []
    text_blobs: list[str] = []
    for item in batch_items:
        fname = item["filename"]
        try:
            caption, humor, tags_json, vlm_text = _vlm_fields(
                item["image_bytes"], filename=fname,
            )
            if caption is None and humor is None and tags_json is None and not vlm_text:
                logger.warning(
                    "VLM backfill[%s]: VLM returned no usable fields — counted as error.",
                    fname,
                )
                errors += 1
                continue
            merged = _merge_ocr_with_vlm_text(item.get("ocr_text", ""), vlm_text or "")
            ok = db.update_meme_vlm(
                filename=fname,
                caption=caption,
                humor_explain=humor,
                tags=tags_json,
                ocr_text=merged if vlm_text else None,
            )
            if ok:
                updated += 1
                refreshed = db.get_meme_by_filename(fname)
                if refreshed and refreshed.get("text_for_embedding"):
                    text_blob_filenames.append(fname)
                    text_blobs.append(refreshed["text_for_embedding"])
            else:
                logger.warning(
                    "VLM backfill[%s]: db.update_meme_vlm returned False "
                    "(row not found or unchanged) — counted as error.",
                    fname,
                )
                errors += 1
        except Exception as e:
            logger.warning(
                "VLM backfill[%s] crashed: %s", fname, e, exc_info=True,
            )
            errors += 1

    if text_blob_filenames:
        try:
            _embed_texts_for_filenames(text_blob_filenames, text_blobs)
        except Exception as e:
            logger.error("VLM backfill BGE re-embed failed: %s", e)

    gc.collect()
    logger.info("VLM backfill batch done: %d updated, %d errors", updated, errors)
    return updated, errors


def _vlm_straggler_batch(batch_items: list[dict]) -> tuple[int, int, int]:
    """
    Last-resort pass for memes the standard VLM prompt couldn't caption:
    - tries `describe_meme_compact` (shorter prompt, num_predict=3072, temperature=0)
    - on success → writes caption + text via update_meme_vlm and re-embeds with BGE
    - on failure → writes SKIPPED_CAPTION_SENTINEL so the meme drops out of
      `caption IS NULL OR caption=''` filters and won't be re-tried forever.
    Returns (rescued, marked_skipped, errors).
    """
    if not batch_items:
        return 0, 0, 0

    rescued = 0
    marked = 0
    errors = 0
    text_blob_filenames: list[str] = []
    text_blobs: list[str] = []

    for item in batch_items:
        fname = item["filename"]
        try:
            try:
                result = vlm_engine.describe_meme_compact(
                    item["image_bytes"], filename=fname,
                )
            except vlm_engine.VLMServiceUnavailable as e:
                # Single-meme timeout / 500 / connection blip. Skip this item
                # WITHOUT writing a sentinel (caption stays NULL → next
                # /vlm/stragglers run will pick it up again). Re-raise so the
                # outer except increments `errors` and we move to the next item.
                logger.warning(
                    "Straggler[%s]: Ollama transient failure (%s) — left as NULL, "
                    "will retry on next /vlm/stragglers run.",
                    fname, e,
                )
                raise
            if result and (result.get("caption") or result.get("text")):
                vlm_text = result.get("text") or ""
                merged = _merge_ocr_with_vlm_text(item.get("ocr_text", ""), vlm_text)
                ok = db.update_meme_vlm(
                    filename=fname,
                    caption=result.get("caption") or None,
                    humor_explain=None,
                    tags=None,
                    ocr_text=merged if vlm_text else None,
                )
                if ok:
                    rescued += 1
                    refreshed = db.get_meme_by_filename(fname)
                    if refreshed and refreshed.get("text_for_embedding"):
                        text_blob_filenames.append(fname)
                        text_blobs.append(refreshed["text_for_embedding"])
                else:
                    logger.warning(
                        "VLM straggler[%s]: db.update_meme_vlm returned False",
                        fname,
                    )
                    errors += 1
                continue

            # Compact prompt also failed → mark as skipped so we don't loop on it
            ok = db.update_meme_vlm(
                filename=fname,
                caption=SKIPPED_CAPTION_SENTINEL,
                humor_explain=None,
                tags=None,
                ocr_text=None,
            )
            if ok:
                marked += 1
                logger.warning(
                    "VLM straggler[%s]: marked as '%s' — both prompts failed.",
                    fname, SKIPPED_CAPTION_SENTINEL,
                )
                refreshed = db.get_meme_by_filename(fname)
                if refreshed and refreshed.get("text_for_embedding"):
                    text_blob_filenames.append(fname)
                    text_blobs.append(refreshed["text_for_embedding"])
            else:
                errors += 1
        except Exception as e:
            logger.warning("VLM straggler[%s] crashed: %s", fname, e, exc_info=True)
            errors += 1

    if text_blob_filenames:
        try:
            _embed_texts_for_filenames(text_blob_filenames, text_blobs)
        except Exception as e:
            logger.error("Straggler BGE re-embed failed: %s", e)

    gc.collect()
    logger.info(
        "Straggler batch done: %d rescued, %d marked-skipped, %d errors",
        rescued, marked, errors,
    )
    return rescued, marked, errors


def _text_embed_backfill_batch(batch_items: list[dict]) -> tuple[int, int]:
    """
    Generate / refresh BGE text embeddings for already-indexed memes.
    Each batch item has {filename, text_for_embedding}.
    """
    if not batch_items:
        return 0, 0
    filenames = [it["filename"] for it in batch_items]
    blobs = [it.get("text_for_embedding") or "" for it in batch_items]
    try:
        n = _embed_texts_for_filenames(filenames, blobs)
    except Exception as e:
        logger.error("Text-embed backfill batch failed: %s", e)
        return 0, len(batch_items)
    return n, len(batch_items) - n


# ── Top-level orchestration ──────────────────────────────────────────────────

async def run_text_embed_backfill():
    """
    (Re)compute BGE text embeddings for every meme that has a populated
    text_for_embedding but is missing from the Chroma `text_embeddings`
    collection. Safe to run repeatedly — upsert by filename.
    """
    global text_embed_state

    if text_embed_state["status"] == "running":
        logger.warning("Text-embed backfill already running.")
        return
    if indexing_state["status"] == "running":
        logger.warning("Cannot start text-embed backfill while indexing is running.")
        return

    text_embed_state.update({
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
        rows = db.get_text_for_embedding_rows()
        existing = chroma_store.get_text_filenames()
        # Backfill rows that either have no text_for_embedding stored yet or
        # are missing from the Chroma text collection. We rebuild the canonical
        # text on the fly so the migration is self-healing.
        targets: list[dict] = []
        for r in rows:
            text = (r.get("text_for_embedding") or "").strip()
            if not text:
                text = db.build_text_for_embedding(
                    r.get("ocr_text"), r.get("caption"),
                    r.get("humor_explain"), r.get("tags"),
                )
                if text:
                    db.update_text_for_embedding(r["filename"], text)
            if not text:
                continue
            if r["filename"] not in existing:
                targets.append({"filename": r["filename"], "text_for_embedding": text})
        text_embed_state["total"] = len(targets)
        logger.info("Text-embed backfill: %d memes to embed", len(targets))

        batch: list[dict] = []
        processed_count = 0

        async def _flush():
            nonlocal processed_count
            if not batch:
                return
            updated, errors = await loop.run_in_executor(
                _executor, _text_embed_backfill_batch, list(batch),
            )
            processed_count += len(batch)
            text_embed_state["processed"] = processed_count
            text_embed_state["updated"] += updated
            text_embed_state["errors"] += errors
            batch.clear()
            text_embed_state["elapsed_seconds"] = (
                time.time() - text_embed_state["started_at"]
            )

        for t in targets:
            batch.append(t)
            if len(batch) >= BATCH_SIZE:
                await _flush()
        await _flush()

        text_embed_state["status"] = "done"
        text_embed_state["elapsed_seconds"] = (
            time.time() - text_embed_state["started_at"]
        )
        logger.info(
            "Text-embed backfill complete: %d processed, %d updated, %d errors, %.1fs",
            text_embed_state["processed"], text_embed_state["updated"],
            text_embed_state["errors"], text_embed_state["elapsed_seconds"],
        )
    except Exception as e:
        logger.error("Text-embed backfill failed: %s", e, exc_info=True)
        text_embed_state["status"] = "error"
        text_embed_state["elapsed_seconds"] = (
            time.time() - text_embed_state["started_at"]
        )


async def run_vlm_stragglers():
    """
    Last-resort pass over memes whose caption is still NULL or empty (or that
    were marked with SKIPPED_CAPTION_SENTINEL on a previous run if the user
    re-runs after fixing the model).

    Uses the compact VLM prompt (text + caption only, num_predict=3072,
    temperature=0). On compact-prompt failure the meme is stamped with
    SKIPPED_CAPTION_SENTINEL so it stops appearing in the queue.
    """
    global straggler_state

    if straggler_state["status"] == "running":
        logger.warning("Straggler backfill already running.")
        return
    if (
        indexing_state["status"] == "running"
        or vlm_state["status"] == "running"
        or reocr_state["status"] == "running"
    ):
        logger.warning("Cannot start straggler backfill while another job is running.")
        return

    straggler_state.update({
        "status": "running",
        "total": 0,
        "processed": 0,
        "updated": 0,
        "marked_skipped": 0,
        "errors": 0,
        "started_at": time.time(),
        "elapsed_seconds": 0,
    })

    loop = asyncio.get_event_loop()

    try:
        missing = db.get_filenames_missing_vlm()
        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
        files_on_disk = {
            e.name: e for e in MEMES_DIR.iterdir()
            if e.is_file() and e.suffix.lower() in valid_ext
        } if MEMES_DIR.exists() else {}

        targets = [files_on_disk[name] for name in missing if name in files_on_disk]
        straggler_state["total"] = len(targets)
        logger.info("Straggler backfill: %d memes without caption", len(targets))

        batch: list[dict] = []
        processed_count = 0

        async def _flush():
            nonlocal processed_count
            if not batch:
                return
            rescued, marked, errors = await loop.run_in_executor(
                _executor, _vlm_straggler_batch, list(batch),
            )
            # Note: if the worker raised VLMServiceUnavailable, the await above
            # re-raises it here and we exit through the outer except block —
            # NO sentinel writes happen for the rest of the queue.
            processed_count += len(batch)
            straggler_state["processed"] = processed_count
            straggler_state["updated"] += rescued
            straggler_state["marked_skipped"] += marked
            straggler_state["errors"] += errors
            batch.clear()
            straggler_state["elapsed_seconds"] = (
                time.time() - straggler_state["started_at"]
            )

        for file_path in targets:
            try:
                image_bytes = file_path.read_bytes()
            except Exception as e:
                logger.error("Cannot read %s: %s", file_path.name, e)
                straggler_state["errors"] += 1
                processed_count += 1
                straggler_state["processed"] = processed_count
                continue

            existing = db.get_meme_by_filename(file_path.name)
            batch.append({
                "filename": file_path.name,
                "image_bytes": image_bytes,
                "ocr_text": (existing or {}).get("ocr_text", "") or "",
            })
            # Compact prompt is heavy (num_predict=3072) — flush often for visible
            # progress in /api/stats while a tiny queue runs.
            if len(batch) >= max(1, BATCH_SIZE // 4):
                await _flush()

        await _flush()

        straggler_state["status"] = "done"
        straggler_state["elapsed_seconds"] = (
            time.time() - straggler_state["started_at"]
        )
        logger.info(
            "Straggler backfill complete: %d processed, %d rescued, "
            "%d marked-skipped, %d errors, %.1fs",
            straggler_state["processed"], straggler_state["updated"],
            straggler_state["marked_skipped"], straggler_state["errors"],
            straggler_state["elapsed_seconds"],
        )

    except vlm_engine.VLMServiceUnavailable as e:
        logger.error(
            "Straggler backfill aborted: Ollama unavailable. "
            "%d already processed (%d rescued, %d marked-skipped). "
            "Restart Ollama and POST /api/index/vlm/stragglers again to retry. "
            "Original error: %s",
            straggler_state["processed"], straggler_state["updated"],
            straggler_state["marked_skipped"], e,
        )
        straggler_state["status"] = "error"
        straggler_state["elapsed_seconds"] = (
            time.time() - straggler_state["started_at"]
        )
    except Exception as e:
        logger.error("Straggler backfill failed: %s", e, exc_info=True)
        straggler_state["status"] = "error"
        straggler_state["elapsed_seconds"] = (
            time.time() - straggler_state["started_at"]
        )


async def run_vlm_backfill(force: bool = False):
    """
    Fill VLM fields (caption/humor/tags) for every already-indexed meme that
    has no caption yet. Embeddings and thumbnails are kept as-is.
    """
    global vlm_state

    if vlm_state["status"] == "running":
        logger.warning("VLM backfill already running.")
        return
    if indexing_state["status"] == "running" or reocr_state["status"] == "running":
        logger.warning("Cannot start VLM backfill while indexing/reocr is running.")
        return
    '''
    if not vlm_engine.is_available():
        vlm_state.update({
            "status": "error",
            "total": 0,
            "processed": 0,
            "updated": 0,
            "errors": 0,
            "started_at": time.time(),
            "elapsed_seconds": 0,
        })
        logger.error("VLM backfill aborted: Ollama/VLM unavailable.")
        return
    '''

    vlm_state.update({
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
        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
        files_on_disk = {
            e.name: e for e in MEMES_DIR.iterdir()
            if e.is_file() and e.suffix.lower() in valid_ext
        } if MEMES_DIR.exists() else {}

        if force:
            targets = list(files_on_disk.values())
        else:
            missing = db.get_filenames_missing_vlm()
            targets = [files_on_disk[name] for name in missing if name in files_on_disk]
        
        if force:
            logger.info("VLM backfill: FORCE mode enabled — reprocessing all %d indexed memes", len(targets))

        vlm_state["total"] = len(targets)
        logger.info("VLM backfill: %d memes to caption", len(targets))

        batch: list[dict] = []
        processed_count = 0

        async def _flush():
            nonlocal processed_count
            if not batch:
                return
            updated, errors = await loop.run_in_executor(
                _executor, _vlm_backfill_batch, list(batch)
            )
            processed_count += len(batch)
            vlm_state["processed"] = processed_count
            vlm_state["updated"] += updated
            vlm_state["errors"] += errors
            batch.clear()
            vlm_state["elapsed_seconds"] = time.time() - vlm_state["started_at"]

        for file_path in targets:
            try:
                image_bytes = file_path.read_bytes()
            except Exception as e:
                logger.error("Cannot read %s: %s", file_path.name, e)
                vlm_state["errors"] += 1
                processed_count += 1
                vlm_state["processed"] = processed_count
                continue

            existing = db.get_meme_by_filename(file_path.name)
            batch.append({
                "filename": file_path.name,
                "image_bytes": image_bytes,
                "ocr_text": (existing or {}).get("ocr_text", "") or "",
            })
            # VLM is sequential and slow — keep the flush batch small so the
            # progress counter updates often.
            if len(batch) >= max(1, BATCH_SIZE // 4):
                await _flush()

        await _flush()

        vlm_state["status"] = "done"
        vlm_state["elapsed_seconds"] = time.time() - vlm_state["started_at"]
        logger.info(
            "VLM backfill complete: %d processed, %d updated, %d errors, %.1fs",
            vlm_state["processed"], vlm_state["updated"], vlm_state["errors"],
            vlm_state["elapsed_seconds"],
        )

    except Exception as e:
        logger.error("VLM backfill failed: %s", e, exc_info=True)
        vlm_state["status"] = "error"
        vlm_state["elapsed_seconds"] = time.time() - vlm_state["started_at"]


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
        chroma_image_filenames = chroma_store.get_image_filenames()
        chroma_text_filenames = chroma_store.get_text_filenames()
        logger.info(
            "Already indexed: %d in SQLite, %d image embeddings, %d text embeddings",
            len(sqlite_filenames), len(chroma_image_filenames), len(chroma_text_filenames),
        )

        valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
        files_to_process = [
            e for e in MEMES_DIR.iterdir()
            if e.is_file() and e.suffix.lower() in valid_ext
        ] if MEMES_DIR.exists() else []

        indexing_state["total"] = len(files_to_process)
        logger.info("Found %d image files to process", len(files_to_process))

        # Three queues:
        #   full_batch  — new files: need OCR + thumbnail + embedding + VLM
        #   embed_batch — in SQLite but missing image embedding (embed-only)
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
            in_image = filename in chroma_image_filenames
            in_text = filename in chroma_text_filenames

            if in_sqlite and in_image and in_text:
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

            if in_sqlite and (not in_image or not in_text):
                # Has OCR/thumbnail but missing one of the embeddings — fast path
                existing = db.get_meme_by_filename(filename) or {}
                # Ensure text_for_embedding exists; recompute if not.
                text_blob = (existing.get("text_for_embedding") or "").strip()
                if not text_blob:
                    text_blob = db.build_text_for_embedding(
                        existing.get("ocr_text"), existing.get("caption"),
                        existing.get("humor_explain"), existing.get("tags"),
                    )
                    if text_blob:
                        db.update_text_for_embedding(filename, text_blob)
                embed_batch.append({
                    "filename": filename,
                    "image_bytes": image_bytes,
                    "text_for_embedding": text_blob,
                })
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
