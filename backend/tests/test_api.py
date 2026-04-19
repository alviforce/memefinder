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

    # Isolated DB
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    database.DB_PATH = db_path
    database._local = threading.local()

    # Mock model loaders so startup doesn't download anything
    with patch.object(clip_engine, "load_models", return_value=None), \
         patch.object(ocr_engine, "load_model", return_value=None):
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
    assert "indexing" in data


def test_stats_empty_db(client):
    data = client.get("/api/stats").json()
    assert data["meme_count"] == 0


# ── /api/search ───────────────────────────────────────────────────────────────

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
         patch("chroma_store.search", return_value=[]) as mock_search:
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
         patch("chroma_store.search", return_value=matches):
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


# ── /api/index/resync ─────────────────────────────────────────────────────────

def test_resync_no_drift(client):
    """When SQLite and ChromaDB are in sync, resync removes nothing."""
    with patch("chroma_store.get_indexed_filenames", return_value=set()), \
         patch("database.get_indexed_filenames", return_value=set()):
        resp = client.post("/api/index/resync")
    assert resp.status_code == 200
    data = resp.json()
    assert data["stale_removed"] == 0


def test_resync_removes_stale_chroma_entries(client):
    """Entries in ChromaDB but not in SQLite should be cleaned up."""
    with patch("chroma_store.get_indexed_filenames", return_value={"ghost.jpg", "real.jpg"}), \
         patch("database.get_indexed_filenames", return_value={"real.jpg"}), \
         patch("chroma_store.delete_embeddings") as mock_delete:
        resp = client.post("/api/index/resync")

    assert resp.status_code == 200
    assert resp.json()["stale_removed"] == 1
    mock_delete.assert_called_once_with(["ghost.jpg"])
