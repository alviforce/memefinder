"""
MemeFinder — SQLite database layer
Stores meme metadata, OCR text (FTS5), and base64 thumbnails.
"""
import sqlite3
import threading
from config import DB_PATH


_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection with WAL mode."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


def init_db():
    """Create tables and FTS5 index if they don't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memes (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            filename         TEXT UNIQUE NOT NULL,
            ocr_text         TEXT,
            thumbnail_base64 TEXT NOT NULL,
            indexed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memes_filename ON memes(filename)
    """)
    # FTS5 virtual table — keeps in sync with memes via triggers
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memes_fts
        USING fts5(filename UNINDEXED, ocr_text, content=memes, content_rowid=id)
    """)
    # Triggers to keep FTS in sync with memes
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memes_ai AFTER INSERT ON memes BEGIN
            INSERT INTO memes_fts(rowid, filename, ocr_text)
            VALUES (new.id, new.filename, new.ocr_text);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memes_ad AFTER DELETE ON memes BEGIN
            INSERT INTO memes_fts(memes_fts, rowid, filename, ocr_text)
            VALUES ('delete', old.id, old.filename, old.ocr_text);
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memes_au AFTER UPDATE ON memes BEGIN
            INSERT INTO memes_fts(memes_fts, rowid, filename, ocr_text)
            VALUES ('delete', old.id, old.filename, old.ocr_text);
            INSERT INTO memes_fts(rowid, filename, ocr_text)
            VALUES (new.id, new.filename, new.ocr_text);
        END
    """)
    conn.commit()


def insert_meme(filename: str, ocr_text: str, thumbnail_base64: str) -> int:
    """Insert a meme. Returns new row id, or 0 if filename already exists."""
    conn = _get_conn()
    cursor = conn.execute(
        "INSERT OR IGNORE INTO memes (filename, ocr_text, thumbnail_base64) VALUES (?, ?, ?)",
        (filename, ocr_text, thumbnail_base64),
    )
    conn.commit()
    return cursor.lastrowid


def delete_meme(filename: str) -> bool:
    """Delete a meme by filename. Returns True if a row was deleted."""
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM memes WHERE filename = ?", (filename,))
    conn.commit()
    return cursor.rowcount > 0


def exists(filename: str) -> bool:
    """Check if a filename is already indexed."""
    conn = _get_conn()
    row = conn.execute("SELECT 1 FROM memes WHERE filename = ?", (filename,)).fetchone()
    return row is not None


def get_indexed_filenames() -> set[str]:
    """Return all indexed filenames."""
    conn = _get_conn()
    rows = conn.execute("SELECT filename FROM memes").fetchall()
    return {row["filename"] for row in rows}


def search_by_text(query: str, limit: int = 30, offset: int = 0) -> list[dict]:
    """
    Full-text search via FTS5 (BM25 ranking).
    Falls back to LIKE if the FTS table is empty or query is not FTS-safe.
    """
    conn = _get_conn()

    # Escape FTS5 special chars so user input is treated as literal text
    safe_query = _fts_escape(query)

    try:
        rows = conn.execute(
            """
            SELECT m.id, m.filename, m.ocr_text, m.thumbnail_base64
            FROM memes_fts f
            JOIN memes m ON m.id = f.rowid
            WHERE memes_fts MATCH ?
            ORDER BY bm25(memes_fts)
            LIMIT ? OFFSET ?
            """,
            (safe_query, limit, offset),
        ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        # Graceful fallback if FTS index is corrupt / query still invalid
        rows = conn.execute(
            "SELECT id, filename, ocr_text, thumbnail_base64 FROM memes "
            "WHERE ocr_text LIKE ? LIMIT ? OFFSET ?",
            (f"%{query}%", limit, offset),
        ).fetchall()
        return [dict(row) for row in rows]


def _fts_escape(query: str) -> str:
    """Wrap each token in double quotes so FTS5 treats them as phrase literals."""
    tokens = query.split()
    if not tokens:
        return '""'
    return " ".join(f'"{t.replace(chr(34), "")}"' for t in tokens)


def get_meme_by_id(meme_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, filename, ocr_text, thumbnail_base64, indexed_at FROM memes WHERE id = ?",
        (meme_id,),
    ).fetchone()
    return dict(row) if row else None


def get_meme_by_filename(filename: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, filename, ocr_text, thumbnail_base64, indexed_at FROM memes WHERE filename = ?",
        (filename,),
    ).fetchone()
    return dict(row) if row else None


def get_memes_by_filenames(filenames: list[str]) -> dict[str, dict]:
    """Batch fetch by filenames. Returns dict keyed by filename."""
    if not filenames:
        return {}
    conn = _get_conn()
    placeholders = ",".join("?" for _ in filenames)
    rows = conn.execute(
        f"SELECT id, filename, ocr_text, thumbnail_base64, indexed_at "
        f"FROM memes WHERE filename IN ({placeholders})",
        filenames,
    ).fetchall()
    return {row["filename"]: dict(row) for row in rows}


def get_memes_by_ids(meme_ids: list[int]) -> dict[int, dict]:
    """Batch fetch by IDs. Returns dict keyed by id."""
    if not meme_ids:
        return {}
    conn = _get_conn()
    placeholders = ",".join("?" for _ in meme_ids)
    rows = conn.execute(
        f"SELECT id, filename, ocr_text, thumbnail_base64 FROM memes WHERE id IN ({placeholders})",
        meme_ids,
    ).fetchall()
    return {row["id"]: dict(row) for row in rows}


def get_meme_count() -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM memes").fetchone()
    return row["cnt"]
