"""
MemeFinder — ChromaDB vector store

Two collections, both keyed by filename:
  - `meme_embeddings`  — CLIP image embeddings (dim=512, cosine).
  - `text_embeddings`  — BGE-M3 dense text embeddings (dim=1024, cosine) over
                         the canonical OCR+caption+humor+tags string.

The legacy `add_embedding` / `search` / `get_indexed_filenames` / `get_count`
functions still operate on the image collection so existing callers (indexer,
resync, /api/stats) keep working untouched. New `*_text_*` helpers expose the
second collection. `*_image_*` aliases mirror them for clarity in new code.
"""
import logging
import chromadb
from config import CHROMA_DIR

logger = logging.getLogger(__name__)

_client = None
_image_collection = None
_text_collection = None

IMAGE_COLLECTION_NAME = "meme_embeddings"   # historical name — kept for compat
TEXT_COLLECTION_NAME = "text_embeddings"


# ── Lazy initialization ──────────────────────────────────────────────────────

def _get_client():
    global _client
    if _client is None:
        logger.info("Initializing ChromaDB at: %s", CHROMA_DIR)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def _get_image_collection():
    global _image_collection
    if _image_collection is None:
        _image_collection = _get_client().get_or_create_collection(
            name=IMAGE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready. Count: %d",
            IMAGE_COLLECTION_NAME, _image_collection.count(),
        )
    return _image_collection


def _get_text_collection():
    global _text_collection
    if _text_collection is None:
        _text_collection = _get_client().get_or_create_collection(
            name=TEXT_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready. Count: %d",
            TEXT_COLLECTION_NAME, _text_collection.count(),
        )
    return _text_collection


# ── Image collection (CLIP) ──────────────────────────────────────────────────

def add_embedding(filename: str, embedding: list[float], metadata: dict | None = None):
    """Add a single image embedding to ChromaDB. (Legacy alias.)"""
    add_image_embedding(filename, embedding, metadata)


def add_image_embedding(filename: str, embedding: list[float], metadata: dict | None = None):
    collection = _get_image_collection()
    collection.upsert(
        ids=[filename],
        embeddings=[embedding],
        metadatas=[metadata or {}],
    )


def add_embeddings_batch(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict] | None = None,
):
    """Batch upsert image embeddings. (Legacy alias.)"""
    add_image_embeddings_batch(ids, embeddings, metadatas)


def add_image_embeddings_batch(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict] | None = None,
):
    if not ids:
        return
    collection = _get_image_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas or [{} for _ in ids],
    )


def search(query_embedding: list[float], n_results: int = 30) -> list[dict]:
    """Search image collection. (Legacy alias for `image_search`.)"""
    return image_search(query_embedding, n_results)


def image_search(query_embedding: list[float], n_results: int = 30) -> list[dict]:
    """
    Search for nearest image embeddings to the query vector.
    Returns list of {filename, distance} sorted by ascending distance.
    """
    collection = _get_image_collection()
    if collection.count() == 0:
        return []
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
    )
    return _format_query_result(results)


def get_indexed_filenames() -> set[str]:
    """Filenames in the IMAGE collection. (Legacy alias.)"""
    return get_image_filenames()


def get_image_filenames() -> set[str]:
    collection = _get_image_collection()
    if collection.count() == 0:
        return set()
    return set(collection.get(include=[])["ids"])


def delete_embeddings(filenames: list[str]):
    """Delete image embeddings. (Legacy alias.)"""
    delete_image_embeddings(filenames)


def delete_image_embeddings(filenames: list[str]):
    if not filenames:
        return
    _get_image_collection().delete(ids=filenames)


def get_count() -> int:
    """Count of image embeddings. (Legacy alias.)"""
    return get_image_count()


def get_image_count() -> int:
    return _get_image_collection().count()


# ── Text collection (BGE-M3) ─────────────────────────────────────────────────

def add_text_embedding(filename: str, embedding: list[float], metadata: dict | None = None):
    if not embedding:
        return
    collection = _get_text_collection()
    collection.upsert(
        ids=[filename],
        embeddings=[embedding],
        metadatas=[metadata or {}],
    )


def add_text_embeddings_batch(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict] | None = None,
):
    if not ids:
        return
    # Filter out empty embeddings — Chroma rejects zero-length vectors
    paired = [
        (i, e, m) for i, e, m in zip(
            ids, embeddings, metadatas or [{} for _ in ids],
        ) if e
    ]
    if not paired:
        return
    f_ids, f_emb, f_meta = zip(*paired)
    _get_text_collection().upsert(
        ids=list(f_ids),
        embeddings=list(f_emb),
        metadatas=list(f_meta),
    )


def text_search(query_embedding: list[float], top_k: int = 30) -> list[dict]:
    """
    Search the text-embedding collection. Returns list of {filename, distance}
    sorted by ascending cosine distance (lower = more similar).
    """
    collection = _get_text_collection()
    if collection.count() == 0 or not query_embedding:
        return []
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )
    return _format_query_result(results)


def get_text_filenames() -> set[str]:
    collection = _get_text_collection()
    if collection.count() == 0:
        return set()
    return set(collection.get(include=[])["ids"])


def delete_text_embeddings(filenames: list[str]):
    if not filenames:
        return
    _get_text_collection().delete(ids=filenames)


def get_text_count() -> int:
    return _get_text_collection().count()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_query_result(results: dict) -> list[dict]:
    """Flatten a Chroma query response (which is single-batched) into dicts."""
    items = []
    if not results.get("ids") or not results["ids"][0]:
        return items
    for i, fname in enumerate(results["ids"][0]):
        distance = (
            results["distances"][0][i] if results.get("distances") else 0.0
        )
        items.append({"filename": fname, "distance": distance})
    return items
