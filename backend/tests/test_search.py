"""
Tests for Phase 3 hybrid search:
  - Reciprocal Rank Fusion (no I/O, pure algorithm).
  - hybrid_search orchestration with mocked retrievers.
  - text_for_embedding canonical builder + DB integration.
"""
import asyncio
import pytest
from unittest.mock import patch


# ── RRF unit tests ───────────────────────────────────────────────────────────

def test_rrf_basic_combines_two_lists():
    from search_service import reciprocal_rank_fusion
    list_a = [{"filename": "a.jpg"}, {"filename": "b.jpg"}, {"filename": "c.jpg"}]
    list_b = [{"filename": "b.jpg"}, {"filename": "a.jpg"}, {"filename": "d.jpg"}]
    fused = reciprocal_rank_fusion([list_a, list_b])
    fnames = [r["filename"] for r in fused]
    # Items in BOTH lists must beat single-list items.
    assert fnames[0] in {"a.jpg", "b.jpg"}
    assert fnames[1] in {"a.jpg", "b.jpg"}
    assert set(fnames[2:]) == {"c.jpg", "d.jpg"}


def test_rrf_skips_empty_lists():
    from search_service import reciprocal_rank_fusion
    fused = reciprocal_rank_fusion([[], [{"filename": "x.jpg"}], []])
    assert len(fused) == 1
    assert fused[0]["filename"] == "x.jpg"


def test_rrf_records_sources():
    from search_service import reciprocal_rank_fusion
    fused = reciprocal_rank_fusion([
        [{"filename": "a.jpg"}],
        [{"filename": "a.jpg"}, {"filename": "b.jpg"}],
    ])
    by_name = {r["filename"]: r for r in fused}
    assert set(by_name["a.jpg"]["sources"]) == {"r0", "r1"}
    assert by_name["b.jpg"]["sources"] == ["r1"]


def test_rrf_weights_boost_retriever():
    from search_service import reciprocal_rank_fusion
    list_a = [{"filename": "a.jpg"}]      # boosted retriever
    list_b = [{"filename": "b.jpg"}]      # un-boosted
    fused = reciprocal_rank_fusion([list_a, list_b], weights=[5.0, 1.0])
    assert fused[0]["filename"] == "a.jpg"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_rrf_handles_filenameless_items():
    from search_service import reciprocal_rank_fusion
    fused = reciprocal_rank_fusion([
        [{"foo": "bar"}, {"filename": "ok.jpg"}],
    ])
    assert len(fused) == 1
    assert fused[0]["filename"] == "ok.jpg"


# ── hybrid_search orchestration ──────────────────────────────────────────────

def test_hybrid_search_text_mode_only_runs_text_retrievers(tmp_db, sample_thumbnail):
    import search_service
    tmp_db.insert_meme("a.jpg", "кот", sample_thumbnail, caption="смешной кот")
    tmp_db.insert_meme("b.jpg", "пес", sample_thumbnail, caption="смешной пёс")

    with patch("search_service.text_embedder.encode_one", return_value=[0.1] * 8), \
         patch("search_service.chroma_store.text_search",
               return_value=[{"filename": "a.jpg", "distance": 0.1}]), \
         patch("search_service.clip_engine.encode_text") as mock_clip_text, \
         patch("search_service.chroma_store.image_search") as mock_image_search:
        result = asyncio.run(
            search_service.hybrid_search(query="кот", mode="text")
        )

    # CLIP must NOT have been called in text-only mode.
    assert mock_clip_text.call_count == 0
    assert mock_image_search.call_count == 0
    assert result["mode"] == "text"
    assert "fts" in result["retrievers"]
    assert "text_vector" in result["retrievers"]
    fnames = [r["filename"] for r in result["results"]]
    assert "a.jpg" in fnames


def test_hybrid_search_all_mode_runs_text_and_image(tmp_db, sample_thumbnail):
    import search_service
    tmp_db.insert_meme("a.jpg", "ничего", sample_thumbnail)

    with patch("search_service.text_embedder.encode_one", return_value=[0.1] * 8), \
         patch("search_service.chroma_store.text_search",
               return_value=[{"filename": "a.jpg", "distance": 0.2}]), \
         patch("search_service.clip_engine.encode_text", return_value=[0.2] * 8), \
         patch("search_service.chroma_store.image_search",
               return_value=[{"filename": "a.jpg", "distance": 0.3}]):
        result = asyncio.run(
            search_service.hybrid_search(query="кот", mode="all")
        )

    assert "fts" in result["retrievers"]
    assert "text_vector" in result["retrievers"]
    assert "image_text" in result["retrievers"]


def test_hybrid_search_image_mode_uses_uploaded_image(tmp_db, sample_thumbnail):
    import io
    import search_service
    from PIL import Image

    tmp_db.insert_meme("a.jpg", "", sample_thumbnail)

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    with patch("search_service.clip_engine.encode_image",
               return_value=[0.5] * 8) as mock_enc_img, \
         patch("search_service.chroma_store.image_search",
               return_value=[{"filename": "a.jpg", "distance": 0.1}]) as mock_img_search:
        result = asyncio.run(
            search_service.hybrid_search(image_bytes=image_bytes, mode="image")
        )

    mock_enc_img.assert_called_once()
    mock_img_search.assert_called_once()
    assert "image_image" in result["retrievers"]
    assert result["results"][0]["filename"] == "a.jpg"


def test_hybrid_search_no_query_no_image_returns_empty():
    import search_service
    result = asyncio.run(search_service.hybrid_search(mode="all"))
    assert result["results"] == []
    assert result["retrievers"] == []


def test_hybrid_search_drops_filenames_missing_from_db(tmp_db, sample_thumbnail):
    """A filename returned by a vector store but absent from SQLite must not
    appear in the response (drift). We just skip it."""
    import search_service
    tmp_db.insert_meme("real.jpg", "real meme", sample_thumbnail)

    with patch("search_service.text_embedder.encode_one", return_value=[0.1] * 8), \
         patch("search_service.chroma_store.text_search", return_value=[
             {"filename": "ghost.jpg", "distance": 0.1},
             {"filename": "real.jpg", "distance": 0.2},
         ]):
        result = asyncio.run(
            search_service.hybrid_search(query="meme", mode="text")
        )

    fnames = [r["filename"] for r in result["results"]]
    assert "ghost.jpg" not in fnames
    assert "real.jpg" in fnames


def test_hybrid_search_paginates(tmp_db, sample_thumbnail):
    import search_service
    for i in range(10):
        tmp_db.insert_meme(f"m{i}.jpg", f"text {i}", sample_thumbnail)

    fts_results = [{"filename": f"m{i}.jpg"} for i in range(10)]
    with patch("search_service.text_embedder.encode_one", return_value=[]), \
         patch("search_service.db.search_by_text",
               return_value=[{"filename": fn["filename"]} for fn in fts_results]):
        page1 = asyncio.run(
            search_service.hybrid_search(query="text", mode="text", top_k=4, offset=0)
        )
        page2 = asyncio.run(
            search_service.hybrid_search(query="text", mode="text", top_k=4, offset=4)
        )
    assert len(page1["results"]) == 4
    assert len(page2["results"]) == 4
    p1 = {r["filename"] for r in page1["results"]}
    p2 = {r["filename"] for r in page2["results"]}
    assert p1.isdisjoint(p2)


# ── Database canonical text builder ──────────────────────────────────────────

def test_build_text_for_embedding_concatenates_fields():
    from database import build_text_for_embedding
    text = build_text_for_embedding(
        ocr_text="HELLO WORLD",
        caption="кот сидит на диване",
        humor_explain="отсылка к интернет-мему 2010-х",
        tags='["кот", "мем", "винтаж"]',
    )
    assert "HELLO WORLD" in text
    assert "кот сидит на диване" in text
    assert "отсылка к интернет-мему 2010-х" in text
    assert "Теги:" in text
    assert "кот" in text


def test_build_text_for_embedding_handles_empty_inputs():
    from database import build_text_for_embedding
    assert build_text_for_embedding(None, None, None, None) == ""
    assert build_text_for_embedding("", "", "", "") == ""


def test_build_text_for_embedding_normalizes_plain_tags():
    from database import build_text_for_embedding
    text = build_text_for_embedding("ocr", None, None, "кот, мем, 2025")
    assert "Теги: кот, мем, 2025" in text


def test_insert_meme_auto_populates_text_for_embedding(tmp_db, sample_thumbnail):
    tmp_db.insert_meme(
        "k.jpg", "ocr text", sample_thumbnail,
        caption="caption text", humor_explain="humor text", tags='["tag1"]',
    )
    row = tmp_db.get_meme_by_filename("k.jpg")
    assert row["text_for_embedding"]
    assert "ocr text" in row["text_for_embedding"]
    assert "caption text" in row["text_for_embedding"]
    assert "humor text" in row["text_for_embedding"]
    assert "tag1" in row["text_for_embedding"]


def test_update_meme_vlm_refreshes_text_for_embedding(tmp_db, sample_thumbnail):
    tmp_db.insert_meme("m.jpg", "ocr only", sample_thumbnail)
    before = tmp_db.get_meme_by_filename("m.jpg")
    assert "ocr only" in (before["text_for_embedding"] or "")
    assert "новое описание" not in (before["text_for_embedding"] or "")

    tmp_db.update_meme_vlm(
        "m.jpg",
        caption="новое описание",
        humor_explain="прикол про котов",
        tags='["котики"]',
    )
    after = tmp_db.get_meme_by_filename("m.jpg")
    assert "новое описание" in after["text_for_embedding"]
    assert "котики" in after["text_for_embedding"]


def test_update_ocr_text_refreshes_text_for_embedding(tmp_db, sample_thumbnail):
    tmp_db.insert_meme("z.jpg", "old text", sample_thumbnail, caption="cap")
    tmp_db.update_ocr_text("z.jpg", "new ocr text")
    row = tmp_db.get_meme_by_filename("z.jpg")
    assert "new ocr text" in row["text_for_embedding"]
    assert "old text" not in row["text_for_embedding"]
    # caption survives
    assert "cap" in row["text_for_embedding"]


def test_get_filenames_missing_text_for_embedding(tmp_db, sample_thumbnail):
    tmp_db.insert_meme("has.jpg", "ocr", sample_thumbnail)
    # Manually wipe text_for_embedding to simulate pre-Phase-3 row.
    conn = tmp_db._get_conn()
    conn.execute("UPDATE memes SET text_for_embedding = NULL WHERE filename = ?", ("has.jpg",))
    conn.commit()

    tmp_db.insert_meme("ok.jpg", "ocr", sample_thumbnail, caption="cap")
    missing = tmp_db.get_filenames_missing_text_for_embedding()
    assert "has.jpg" in missing
    assert "ok.jpg" not in missing
