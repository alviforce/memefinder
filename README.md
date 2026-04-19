## Project: MemeFinder

A local meme search engine that indexes images from a `memes/` folder using OCR (EasyOCR) and semantic embeddings (CLIP), with a React frontend and FastAPI backend.

## Directory Layout

```
memefinder-no-api/
├── memes/              # Drop images here (jpg/jpeg/png/webp)
├── backend/
│   ├── main.py         # FastAPI app, API routes
│   ├── config.py       # All constants (paths, model names, batch size)
│   ├── database.py     # SQLite layer with FTS5 full-text search
│   ├── chroma_store.py # ChromaDB vector store wrapper
│   ├── clip_engine.py  # CLIP model (two-model design: multilingual text + image)
│   ├── ocr_engine.py   # EasyOCR wrapper
│   ├── indexer.py      # Async indexing pipeline
│   └── tests/
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── hooks/useSearch.js   # All API calls + infinite scroll logic
    │   └── components/
    └── vite.config.js           # Proxies /api → http://127.0.0.1:8000
```

## Running the App

**Backend** (from `backend/`):
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend** (from `frontend/`):
```bash
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`. Vite proxies all `/api` requests to the backend at `http://127.0.0.1:8000`.

## Tests

Run from `backend/`:
```bash
pytest                        # all tests
pytest -m "not slow"          # skip ML model tests
pytest tests/test_database.py # single file
```

## Architecture

**Dual-store design:** Every indexed meme lives in two stores that must stay in sync:
- **SQLite** (`memefinder.db`) — stores filename, OCR text, and base64 thumbnail. OCR text is indexed with FTS5 (BM25 ranking). Thread-local connections with WAL mode.
- **ChromaDB** (`chroma_data/`) — stores CLIP image embeddings for semantic search. Uses cosine similarity. IDs are filenames.

**Two-model CLIP setup:** Text queries use `clip-ViT-B-32-multilingual-v1` (supports ru/en), image encoding uses `clip-ViT-B-32`. Both produce embeddings in the same vector space via knowledge distillation, so cross-modal similarity is valid.

**Indexing pipeline** (`indexer.py`): Async, with CPU/GPU work offloaded to a single-threaded `ThreadPoolExecutor` (ML models are not thread-safe). Distinguishes three file states:
1. In SQLite + ChromaDB → skip
2. In SQLite only → embed-only fast path (no re-OCR)
3. New → full pipeline (thumbnail + OCR + embedding)

**Search flow:**
- OCR mode: FTS5 `MATCH` query on `memes_fts`, falls back to `LIKE` on error
- CLIP mode: encode text → ChromaDB vector query → batch-fetch metadata from SQLite

**`/api/index/resync`** detects and fixes divergence between the two stores (removes ChromaDB entries with no corresponding SQLite row).

## Config

All tunable values are in `backend/config.py`. Key ones:
- `MEMES_DIR` — source image folder (default: `../memes/` relative to backend)
- `BATCH_SIZE` — images per ML batch (default: 16)
- `INDEX_LIMIT` — 0 means no limit
- `DEVICE` — auto-detected (`cuda` if available, else `cpu`)
- `OCR_LANGUAGES` — `["ru", "en"]`
