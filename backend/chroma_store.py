"""
MemeFinder v2.0 — ChromaDB vector store
Stores and queries meme image embeddings for semantic search.
ID format: "{filename}"
"""
import logging
import chromadb
from config import CHROMA_DIR

logger = logging.getLogger(__name__)

_client = None
_collection = None

COLLECTION_NAME = "meme_embeddings"


def _get_collection():
    """Lazy-init ChromaDB persistent client and collection."""
    global _client, _collection
    if _collection is None:
        logger.info("Initializing ChromaDB at: %s", CHROMA_DIR)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready. Count: %d",
            COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


def add_embedding(filename: str, embedding: list[float], metadata: dict | None = None):
    """Add a single image embedding to ChromaDB."""
    collection = _get_collection()
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
    """Batch upsert multiple embeddings at once."""
    if not ids:
        return
    collection = _get_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas or [{} for _ in ids],
    )


def search(query_embedding: list[float], n_results: int = 30) -> list[dict]:
    """
    Search for nearest image embeddings to the query vector.
    Returns list of {filename, distance}.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
    )

    items = []
    for i in range(len(results["ids"][0])):
        filename = results["ids"][0][i]
        items.append(
            {
                "filename": filename,
                "distance": results["distances"][0][i] if results["distances"] else 0,
            }
        )
    return items


def get_indexed_filenames() -> set[str]:
    """Return all filenames that have an embedding in ChromaDB."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return set()
    result = collection.get(include=[])  # fetch only ids, no embeddings
    return set(result["ids"])


def delete_embeddings(filenames: list[str]):
    """Delete embeddings for the given filenames."""
    if not filenames:
        return
    collection = _get_collection()
    collection.delete(ids=filenames)


def get_count() -> int:
    """Return total number of embeddings in the collection."""
    collection = _get_collection()
    return collection.count()
