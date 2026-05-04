"""
Tests for API endpoints (main.py).
ML models are mocked — no GPU/download required.
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    TestClient with:
    - isolated SQLite DB
    - mocked ML models (no actual loading)
    """
    import threading
    import database
    import clip_engine
    import ocr_engine
    import text_embedder
    import vlm_engine

    # Isolated DB
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    database.DB_PATH = db_path
    database._local = threading.local()

    # Mock model loaders so startup doesn't download anything
    with patch.object(clip_engine, "load_models", return_value=None), \
         patch.object(ocr_engine, "load_model", return_value=None), \
         patch.object(text_embedder, "load_model", return_value=None), \
         patch.object(vlm_engine, "load_model", return_value=None):
        from main import app
        with TestClient(app) as c:
            yield c


# ── /api/stats ────────────────────────────────────────────────────────────────

def test_stats_returns_200(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200


def test_stats_structure(client):
    data = client.get("/api/stats").json()
    assert "meme_count" in data
    assert "embedding_count" in data
    assert "text_embedding_count" in data
    assert "indexing" in data
    assert "text_embed" in data


def test_stats_empty_db(client):
    data = client.get("/api/stats").json()
    assert data["meme_count"] == 0


# ── GET /api/search (legacy single-retriever) ────────────────────────────────

def test_search_requires_query(client):
    resp = client.get("/api/search")
    assert resp.status_code == 422


def test_search_rejects_empty_query(client):
    resp = client.get("/api/search?q=")
    assert resp.status_code == 422


def test_search_invalid_mode(client):
    resp = client.get("/api/search?q=test&mode=invalid")
    assert resp.status_code == 422


def test_search_ocr_empty_db(client):
    resp = client.get("/api/search?q=кот&mode=ocr")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "ocr"
    assert data["results"] == []


def test_search_clip_calls_encode(client):
    dummy_emb = [0.1] * 512
    with patch("clip_engine.encode_text", return_value=dummy_emb) as mock_enc, \
         patch("chroma_store.image_search", return_value=[]) as mock_search:
        resp = client.get("/api/search?q=тест&mode=clip")
    assert resp.status_code == 200
    mock_enc.assert_called_once_with("тест")
    mock_search.assert_called_once()


def test_search_clip_returns_results(client, sample_thumbnail):
    import database
    database.init_db()
    database.insert_meme("funny.jpg", "смешной кот", sample_thumbnail)

    dummy_emb = [0.1] * 512
    matches = [{"filename": "funny.jpg", "distance": 0.1}]

    with patch("clip_engine.encode_text", return_value=dummy_emb), \
         patch("chroma_store.image_search", return_value=matches):
        resp = client.get("/api/search?q=кот&mode=clip")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["filename"] == "funny.jpg"
    assert data["results"][0]["score"] == pytest.approx(0.9, abs=0.001)


def test_search_clip_empty_embedding(client):
    with patch("clip_engine.encode_text", return_value=[]):
        resp = client.get("/api/search?q=тест&mode=clip")
    assert resp.status_code == 500


# ── POST /api/search (Phase 3 hybrid) ────────────────────────────────────────

def test_hybrid_search_requires_input(client):
    """Reject requests with neither query nor image."""
    resp = client.post("/api/search", data={"mode": "all"})
    assert resp.status_code == 400


def test_hybrid_search_text_only(client, sample_thumbnail):
    import database
    database.init_db()
    database.insert_meme("a.jpg", "кот сидит", sample_thumbnail, caption="смешной кот")

    with patch("text_embedder.encode_one", return_value=[0.1] * 8), \
         patch("chroma_store.text_search",
               return_value=[{"filename": "a.jpg", "distance": 0.1}]):
        resp = client.post("/api/search", data={"query": "кот", "mode": "text"})
    assert resp.status_code == 200
    data = resp.json()
    assert "fts" in data["retrievers"]
    fnames = [r["filename"] for r in data["results"]]
    assert "a.jpg" in fnames


def test_hybrid_search_returns_tags_as_list(client, sample_thumbnail):
    import database
    database.init_db()
    database.insert_meme(
        "tagged.jpg", "ocr", sample_thumbnail,
        caption="cap", tags='["funny", "cat"]',
    )
    with patch("text_embedder.encode_one", return_value=[0.1] * 8), \
         patch("chroma_store.text_search",
               return_value=[{"filename": "tagged.jpg", "distance": 0.1}]):
        resp = client.post("/api/search", data={"query": "cat", "mode": "text"})
    data = resp.json()
    assert resp.status_code == 200
    item = next(r for r in data["results"] if r["filename"] == "tagged.jpg")
    assert item["tags"] == ["funny", "cat"]
    assert item["caption"] == "cap"


# ── /api/image/original/{filename} ───────────────────────────────────────────

def test_image_not_found(client):
    resp = client.get("/api/image/original/nonexistent.jpg")
    assert resp.status_code == 404


def test_image_path_traversal_blocked(client):
    for payload in ["../secret.jpg", "../../etc/passwd", "foo/bar.jpg", "foo\\bar.jpg"]:
        resp = client.get(f"/api/image/original/{payload}")
        assert resp.status_code in (400, 404), f"Expected 400/404 for '{payload}', got {resp.status_code}"


def test_image_serves_file(client, tmp_path):
    from config import MEMES_DIR
    from PIL import Image

    img_path = MEMES_DIR / "test_serve.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    try:
        resp = client.get("/api/image/original/test_serve.jpg")
        assert resp.status_code == 200
        assert "image" in resp.headers["content-type"]
    finally:
        img_path.unlink(missing_ok=True)


# ── /api/index/start ──────────────────────────────────────────────────────────

def test_index_start_returns_200(client):
    with patch("indexer.run_indexing", return_value=None):
        resp = client.post("/api/index/start")
    assert resp.status_code == 200


def test_index_start_rejects_concurrent(client):
    import indexer
    original_state = indexer.indexing_state.copy()
    indexer.indexing_state["status"] = "running"
    try:
        resp = client.post("/api/index/start")
        assert resp.status_code == 200
        assert "Already running" in resp.json()["message"]
    finally:
        indexer.indexing_state.update(original_state)


# ── /api/index/text-embeddings (Phase 3) ─────────────────────────────────────

def test_text_embed_backfill_returns_200(client):
    with patch("indexer.run_text_embed_backfill", return_value=None):
        resp = client.post("/api/index/text-embeddings")
    assert resp.status_code == 200


def test_text_embed_backfill_rejects_concurrent_indexing(client):
    import indexer
    original_state = indexer.indexing_state.copy()
    indexer.indexing_state["status"] = "running"
    try:
        resp = client.post("/api/index/text-embeddings")
        assert resp.status_code == 409
    finally:
        indexer.indexing_state.update(original_state)


# ── /api/index/resync ─────────────────────────────────────────────────────────

def test_resync_no_drift(client):
    """When SQLite and ChromaDB are in sync, resync removes nothing."""
    with patch("chroma_store.get_image_filenames", return_value=set()), \
         patch("chroma_store.get_text_filenames", return_value=set()), \
         patch("database.get_indexed_filenames", return_value=set()):
        resp = client.post("/api/index/resync")
    assert resp.status_code == 200
    data = resp.json()
    assert data["stale_image_removed"] == 0
    assert data["stale_text_removed"] == 0


def test_resync_removes_stale_chroma_entries(client):
    """Entries in ChromaDB but not in SQLite should be cleaned up."""
    with patch("chroma_store.get_image_filenames",
               return_value={"ghost.jpg", "real.jpg"}), \
         patch("chroma_store.get_text_filenames",
               return_value={"ghost-text.jpg", "real.jpg"}), \
         patch("database.get_indexed_filenames", return_value={"real.jpg"}), \
         patch("chroma_store.delete_image_embeddings") as mock_del_img, \
         patch("chroma_store.delete_text_embeddings") as mock_del_txt:
        resp = client.post("/api/index/resync")

    assert resp.status_code == 200
    body = resp.json()
    assert body["stale_image_removed"] == 1
    assert body["stale_text_removed"] == 1
    mock_del_img.assert_called_once_with(["ghost.jpg"])
    mock_del_txt.assert_called_once_with(["ghost-text.jpg"])
