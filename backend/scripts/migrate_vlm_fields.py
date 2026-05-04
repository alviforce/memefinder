"""
Phase 3 migration:
  1. Make sure every meme row has text_for_embedding populated
     (rebuild from ocr_text + caption + humor_explain + tags).
  2. (Re)compute BGE-M3 dense vectors for any meme that has the canonical
     text but is missing from the Chroma `text_embeddings` collection.

Run from the backend directory:
    python -m scripts.migrate_vlm_fields
or:
    python scripts/migrate_vlm_fields.py

Idempotent — safe to run multiple times.
"""
import logging
import sys
import time
from pathlib import Path

# Allow running as a plain script (`python scripts/migrate_vlm_fields.py`)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chroma_store
import database as db
import text_embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase3.migrate")

BATCH = 32


def step_1_backfill_text_field() -> int:
    """Populate text_for_embedding wherever it's missing. Returns rows updated."""
    rows = db.get_text_for_embedding_rows()
    updated = 0
    for r in rows:
        if (r.get("text_for_embedding") or "").strip():
            continue
        text = db.build_text_for_embedding(
            r.get("ocr_text"), r.get("caption"),
            r.get("humor_explain"), r.get("tags"),
        )
        if not text:
            continue  # nothing to embed yet (missing OCR + VLM both)
        if db.update_text_for_embedding(r["filename"], text):
            updated += 1
    logger.info("Step 1: text_for_embedding populated for %d rows", updated)
    return updated


def step_2_backfill_vectors() -> int:
    """Embed every row that has text but is absent from `text_embeddings`."""
    rows = db.get_text_for_embedding_rows()
    existing = chroma_store.get_text_filenames()

    pending: list[tuple[str, str]] = []
    for r in rows:
        text = (r.get("text_for_embedding") or "").strip()
        if not text:
            continue
        if r["filename"] in existing:
            continue
        pending.append((r["filename"], text))

    logger.info("Step 2: %d rows need a BGE vector", len(pending))
    if not pending:
        return 0

    text_embedder.load_model()
    written = 0
    for i in range(0, len(pending), BATCH):
        chunk = pending[i : i + BATCH]
        names, texts = zip(*chunk)
        t0 = time.perf_counter()
        embs = text_embedder.encode(list(texts))
        chroma_store.add_text_embeddings_batch(
            ids=list(names),
            embeddings=embs,
            metadatas=[{"filename": n} for n in names],
        )
        written += sum(1 for e in embs if e)
        logger.info(
            "  batch %d/%d  (%d items, %.1fs)",
            i // BATCH + 1,
            (len(pending) + BATCH - 1) // BATCH,
            len(chunk),
            time.perf_counter() - t0,
        )
    logger.info("Step 2: wrote %d BGE vectors to ChromaDB", written)
    return written


def main() -> int:
    db.init_db()
    logger.info("DB ready: %d memes", db.get_meme_count())

    n1 = step_1_backfill_text_field()
    n2 = step_2_backfill_vectors()

    logger.info(
        "Migration complete. text_for_embedding updated=%d, BGE vectors written=%d",
        n1, n2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
