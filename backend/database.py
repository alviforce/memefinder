"""
MemeFinder — SQLite database layer
Stores meme metadata, OCR text (FTS5), and base64 thumbnails.
"""
import json
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
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            filename            TEXT UNIQUE NOT NULL,
            ocr_text            TEXT,
            thumbnail_base64    TEXT NOT NULL,
            caption             TEXT,
            humor_explain       TEXT,
            tags                TEXT,
            text_for_embedding  TEXT,
            indexed_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memes_filename ON memes(filename)
    """)
    # Migrate older DBs — add columns if missing.
    for col in ("caption", "humor_explain", "tags", "text_for_embedding"):
        try:
            conn.execute(f"ALTER TABLE memes ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass  # already exists
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


# ── Text-for-embedding helpers ───────────────────────────────────────────────

def build_text_for_embedding(
    ocr_text: str | None,
    caption: str | None,
    humor_explain: str | None,
    tags: str | None,
) -> str:
    """
    Compose the canonical text we feed to BGE-M3 for a meme.
    Tags can be a JSON-encoded list (the indexer stores them that way) or a
    plain comma-separated string — both are normalized here.
    """
    parts: list[str] = []
    ocr_text = (ocr_text or "").strip()
    caption = (caption or "").strip()
    humor_explain = (humor_explain or "").strip()

    if ocr_text:
        parts.append(ocr_text)
    if caption:
        parts.append(caption)
    if humor_explain:
        parts.append(humor_explain)

    tag_str = _normalize_tag_string(tags)
    if tag_str:
        parts.append(f"Теги: {tag_str}")

    return "\n\n".join(parts).strip()


def _normalize_tag_string(tags: str | None) -> str:
    """Convert tags (JSON list or plain string) to a comma-separated string."""
    if not tags:
        return ""
    raw = tags.strip()
    if not raw:
        return ""
    # Tags may be stored as a JSON-encoded list (indexer.py uses json.dumps).
    if raw.startswith("["):
        try:
            items = json.loads(raw)
            if isinstance(items, list):
                return ", ".join(str(t).strip() for t in items if str(t).strip())
        except json.JSONDecodeError:
            pass
    return raw


def insert_meme(
    filename: str,
    ocr_text: str,
    thumbnail_base64: str,
    caption: str | None = None,
    humor_explain: str | None = None,
    tags: str | None = None,
    text_for_embedding: str | None = None,
) -> int:
    """Insert a meme. Returns new row id, or 0 if filename already exists."""
    conn = _get_conn()
    if text_for_embedding is None:
        text_for_embedding = build_text_for_embedding(
            ocr_text, caption, humor_explain, tags
        )
    cursor = conn.execute(
        "INSERT OR IGNORE INTO memes "
        "(filename, ocr_text, thumbnail_base64, caption, humor_explain, tags, text_for_embedding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (filename, ocr_text, thumbnail_base64, caption, humor_explain, tags, text_for_embedding),
    )
    conn.commit()
    return cursor.lastrowid


def update_meme_vlm(
    filename: str,
    caption: str | None,
    humor_explain: str | None,
    tags: str | None,
    ocr_text: str | None = None,
) -> bool:
    """
    Update VLM-derived fields for an existing meme. If ocr_text is provided,
    it replaces the stored OCR text (useful when VLM yields better text).
    Always recomputes text_for_embedding so the BGE backfill stays consistent.
    Returns True if a row was updated.
    """
    conn = _get_conn()
    existing = get_meme_by_filename(filename)
    if not existing:
        return False
    final_ocr = ocr_text if ocr_text is not None else existing["ocr_text"]
    new_text_for_embedding = build_text_for_embedding(
        final_ocr, caption, humor_explain, tags
    )
    if ocr_text is not None:
        cursor = conn.execute(
            "UPDATE memes SET caption = ?, humor_explain = ?, tags = ?, "
            "ocr_text = ?, text_for_embedding = ? "
            "WHERE filename = ?",
            (caption, humor_explain, tags, ocr_text, new_text_for_embedding, filename),
        )
    else:
        cursor = conn.execute(
            "UPDATE memes SET caption = ?, humor_explain = ?, tags = ?, "
            "text_for_embedding = ? "
            "WHERE filename = ?",
            (caption, humor_explain, tags, new_text_for_embedding, filename),
        )
    conn.commit()
    return cursor.rowcount > 0


def update_text_for_embedding(filename: str, text_for_embedding: str) -> bool:
    """Persist the canonical text used for BGE embedding (recomputed on demand)."""
    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE memes SET text_for_embedding = ? WHERE filename = ?",
        (text_for_embedding, filename),
    )
    conn.commit()
    return cursor.rowcount > 0


def get_filenames_missing_vlm() -> set[str]:
    """Filenames of memes with no VLM caption yet."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT filename FROM memes WHERE caption IS NULL OR caption = ''"
    ).fetchall()
    return {row["filename"] for row in rows}


def get_filenames_missing_text_for_embedding() -> set[str]:
    """Filenames of memes whose canonical embedding text hasn't been built yet."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT filename FROM memes "
        "WHERE text_for_embedding IS NULL OR text_for_embedding = ''"
    ).fetchall()
    return {row["filename"] for row in rows}


def get_text_for_embedding_rows() -> list[dict]:
    """All rows with their (possibly empty) text_for_embedding field."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT filename, ocr_text, caption, humor_explain, tags, text_for_embedding "
        "FROM memes"
    ).fetchall()
    return [dict(r) for r in rows]


def update_ocr_text(filename: str, ocr_text: str) -> bool:
    """Update OCR text for an existing meme. Also refreshes text_for_embedding."""
    conn = _get_conn()
    existing = get_meme_by_filename(filename)
    if not existing:
        return False
    new_text_for_embedding = build_text_for_embedding(
        ocr_text, existing["caption"], existing["humor_explain"], existing["tags"]
    )
    cursor = conn.execute(
        "UPDATE memes SET ocr_text = ?, text_for_embedding = ? WHERE filename = ?",
        (ocr_text, new_text_for_embedding, filename),
    )
    conn.commit()
    return cursor.rowcount > 0


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
            f"""
            SELECT {_MEME_COLS_PREFIXED}
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
            f"SELECT {_MEME_COLS} FROM memes "
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


_MEME_COLS = (
    "id, filename, ocr_text, thumbnail_base64, "
    "caption, humor_explain, tags, text_for_embedding, indexed_at"
)
# Same columns prefixed with `m.` for joined queries against memes_fts.
_MEME_COLS_PREFIXED = ", ".join(f"m.{c.strip()}" for c in _MEME_COLS.split(","))


def get_meme_by_id(meme_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        f"SELECT {_MEME_COLS} FROM memes WHERE id = ?",
        (meme_id,),
    ).fetchone()
    return dict(row) if row else None


def get_meme_by_filename(filename: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        f"SELECT {_MEME_COLS} FROM memes WHERE filename = ?",
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
        f"SELECT {_MEME_COLS} FROM memes WHERE filename IN ({placeholders})",
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
        f"SELECT {_MEME_COLS} FROM memes WHERE id IN ({placeholders})",
        meme_ids,
    ).fetchall()
    return {row["id"]: dict(row) for row in rows}


def get_meme_count() -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM memes").fetchone()
    return row["cnt"]
