"""
Tests for database.py — SQLite layer.
"""
import pytest


def test_init_creates_table(tmp_db):
    conn = tmp_db._get_conn()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memes'"
    ).fetchone()
    assert row is not None, "Table 'memes' should exist after init_db()"


def test_insert_and_get_by_id(tmp_db, sample_thumbnail):
    meme_id = tmp_db.insert_meme("cat.jpg", "кот на диване", sample_thumbnail)
    assert meme_id > 0

    meme = tmp_db.get_meme_by_id(meme_id)
    assert meme is not None
    assert meme["filename"] == "cat.jpg"
    assert meme["ocr_text"] == "кот на диване"


def test_insert_duplicate_is_ignored(tmp_db, sample_thumbnail):
    tmp_db.insert_meme("dup.jpg", "text", sample_thumbnail)
    tmp_db.insert_meme("dup.jpg", "text2", sample_thumbnail)
    # Second insert is silently ignored — count stays at 1
    assert tmp_db.get_meme_count() == 1
    # Original text is preserved
    meme = tmp_db.get_meme_by_filename("dup.jpg")
    assert meme["ocr_text"] == "text"


def test_get_meme_count(tmp_db, sample_thumbnail):
    assert tmp_db.get_meme_count() == 0
    tmp_db.insert_meme("a.jpg", "", sample_thumbnail)
    tmp_db.insert_meme("b.jpg", "", sample_thumbnail)
    assert tmp_db.get_meme_count() == 2


def test_get_indexed_filenames(tmp_db, sample_thumbnail):
    tmp_db.insert_meme("x.jpg", "", sample_thumbnail)
    tmp_db.insert_meme("y.jpg", "", sample_thumbnail)
    names = tmp_db.get_indexed_filenames()
    assert names == {"x.jpg", "y.jpg"}


def test_search_by_text_finds_match(tmp_db, sample_thumbnail):
    tmp_db.insert_meme("meme1.jpg", "кот смешной", sample_thumbnail)
    tmp_db.insert_meme("meme2.jpg", "собака весёлая", sample_thumbnail)

    results = tmp_db.search_by_text("кот")
    assert len(results) == 1
    assert results[0]["filename"] == "meme1.jpg"


def test_search_by_text_no_match(tmp_db, sample_thumbnail):
    tmp_db.insert_meme("meme.jpg", "кот", sample_thumbnail)
    results = tmp_db.search_by_text("собака")
    assert len(results) == 0


def test_search_by_text_pagination(tmp_db, sample_thumbnail):
    for i in range(5):
        tmp_db.insert_meme(f"meme{i}.jpg", "тест тест", sample_thumbnail)

    page1 = tmp_db.search_by_text("тест", limit=3, offset=0)
    page2 = tmp_db.search_by_text("тест", limit=3, offset=3)
    assert len(page1) == 3
    assert len(page2) == 2
    # No overlap
    ids1 = {r["id"] for r in page1}
    ids2 = {r["id"] for r in page2}
    assert ids1.isdisjoint(ids2)


def test_search_fts_special_chars_dont_crash(tmp_db, sample_thumbnail):
    """FTS5 special chars in user input should be escaped, not raise errors."""
    tmp_db.insert_meme("meme.jpg", "hello world", sample_thumbnail)
    # These would crash raw FTS5 MATCH but should be safely escaped
    for query in ['hello"world', "AND OR NOT", "*test*", ""]:
        results = tmp_db.search_by_text(query)
        assert isinstance(results, list)


def test_delete_meme(tmp_db, sample_thumbnail):
    tmp_db.insert_meme("del.jpg", "to delete", sample_thumbnail)
    assert tmp_db.exists("del.jpg")
    deleted = tmp_db.delete_meme("del.jpg")
    assert deleted is True
    assert not tmp_db.exists("del.jpg")
    # Deleted meme should not appear in search
    results = tmp_db.search_by_text("delete")
    assert len(results) == 0


def test_get_memes_by_filenames(tmp_db, sample_thumbnail):
    tmp_db.insert_meme("a.jpg", "text a", sample_thumbnail)
    tmp_db.insert_meme("b.jpg", "text b", sample_thumbnail)
    tmp_db.insert_meme("c.jpg", "text c", sample_thumbnail)

    result = tmp_db.get_memes_by_filenames(["a.jpg", "c.jpg"])
    assert set(result.keys()) == {"a.jpg", "c.jpg"}
    assert result["a.jpg"]["ocr_text"] == "text a"


def test_get_memes_by_filenames_empty(tmp_db):
    result = tmp_db.get_memes_by_filenames([])
    assert result == {}


def test_get_meme_by_filename(tmp_db, sample_thumbnail):
    tmp_db.insert_meme("find_me.jpg", "some text", sample_thumbnail)
    meme = tmp_db.get_meme_by_filename("find_me.jpg")
    assert meme is not None
    assert meme["filename"] == "find_me.jpg"


def test_get_meme_by_filename_not_found(tmp_db):
    assert tmp_db.get_meme_by_filename("ghost.jpg") is None


def test_exists(tmp_db, sample_thumbnail):
    assert not tmp_db.exists("new.jpg")
    tmp_db.insert_meme("new.jpg", "", sample_thumbnail)
    assert tmp_db.exists("new.jpg")
