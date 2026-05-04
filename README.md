# MemeFinder

A local meme search engine. Indexes a `memes/` folder and finds the right
meme by text, by meaning, or by image — all running locally, no external APIs.

Search is hybrid: every query is dispatched to three retrievers in parallel
(SQLite FTS5 keyword, BGE-M3 semantic, CLIP visual) and the result lists are
merged through **Reciprocal Rank Fusion** into a single ranked feed.

## Features

- **Hybrid search** — FTS5 + BGE-M3 + CLIP, fused via RRF.
- **OCR** through EasyOCR + PaddleOCR ensemble with CLAHE preprocessing for
  stylized meme fonts.
- **VLM captioning** via Ollama (`qwen3-vl-abliterated:8b-instruct`) — caption,
  joke explanation, tags.
- **CLIP** for meaning-based and image-to-image search.
- **BGE-M3** (multilingual, dim=1024) for semantic search over OCR + VLM fields.
- **Infinite scroll** on the frontend, async indexing on the backend.
- **Fully local** — no cloud calls, everything runs on your machine.

## Stack

**Backend:** Python 3.13 / FastAPI / SQLite (FTS5) / ChromaDB / sentence-transformers /
EasyOCR + PaddleOCR / Ollama
**Frontend:** React 18 / Vite

## Quick start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

On first launch the ML models will download (~3 GB total):
- `clip-ViT-B-32-multilingual-v1` (~500 MB)
- `clip-ViT-B-32` (~340 MB)
- `BAAI/bge-m3` (~2.3 GB)
- EasyOCR into `~/.EasyOCR/`, PaddleOCR into `~/.paddleocr/`

### 2. Ollama (for VLM captions)

```bash
ollama pull huihui_ai/qwen3-vl-abliterated:8b-instruct
ollama serve
```

If Ollama isn't running, indexing proceeds without VLM fields (OCR + CLIP only).

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend at `http://localhost:5173`. Vite proxies `/api` to `http://127.0.0.1:8000`.

### 4. Drop memes and index

Put images (`.jpg` / `.jpeg` / `.png` / `.webp`) into the `memes/` folder, open
the frontend, and click **"Start indexing from memes/ folder"**. Progress is
streamed via `/api/stats`.

## Search modes

The top panel exposes three modes:

| Mode | What it does | When to use |
|---|---|---|
| **Hybrid** | FTS5 + BGE-M3 + CLIP, RRF fusion | Default — best out-of-the-box result |
| **By text** | FTS5 + BGE-M3 (no CLIP) | When you trust the wording |
| **By meaning** | CLIP only | When you want pure visual similarity |

The camera button next to the search input loads an image for image-to-image
search. It can be combined with text: "find memes similar to this image and
about cats."

## Architecture

### Storage

Each meme lives in two stores; drift is healed by `/api/index/resync`:

- **SQLite** (`memefinder.db`) — `memes` (filename, ocr_text, thumbnail_base64,
  caption, humor_explain, tags, text_for_embedding) plus `memes_fts` (FTS5
  BM25 over `ocr_text`). Thread-local connections, WAL mode.
- **ChromaDB** (`chroma_data/`) — two collections:
  - `meme_embeddings` — CLIP image embeddings (dim=512, cosine).
  - `text_embeddings` — BGE-M3 dense vectors (dim=1024, cosine) over
    `text_for_embedding = OCR + caption + humor + tags`.

### Indexing pipeline

All CPU/GPU work runs in a single-threaded `ThreadPoolExecutor` so it can't
block the async loop or race on non-thread-safe ML models.

For each file we determine its state:
1. In SQLite + both ChromaDB collections -> **skip**.
2. In SQLite, missing CLIP or BGE vector -> **embed-only** fast path.
3. New -> **full pipeline**:
   - OCR (EasyOCR + PaddleOCR ensemble with CV preprocessing)
   - VLM (Ollama -> JSON with `text`, `caption`, `humor`, `tags`)
   - thumbnail (PIL -> base64)
   - CLIP image embedding
   - BGE-M3 text embedding
   - persist to SQLite + both ChromaDB collections

### Hybrid search

`backend/search_service.py:hybrid_search()`:

1. Retrievers run in parallel through the executor:
   - **FTS5** — top-60 by BM25 over `ocr_text`.
   - **BGE-M3** — encode the query, top-60 cosine in `text_embeddings`.
   - **CLIP** — text->image (text-only query) or image->image (uploaded image),
     top-40 in `meme_embeddings`.
2. **Reciprocal Rank Fusion** combines ranks:
   `score = sum_l(weight_l / (k1 + rank_l * k2))`, k1=60, k2=20.
   Weights: BGE 1.2 (strongest single signal on ru/en), CLIP-text 0.7, others 1.0.
3. Dedupe -> paginate -> batch-fetch metadata from SQLite.
4. The response includes `timing_ms` per stage — handy for tuning.

### VLM captions

Uses `qwen3-vl-abliterated:8b-instruct` through Ollama with `format="json"` for
strict-JSON output `{text, caption, humor, tags}`. To handle truncated JSON or
temperature quirks:
- **One retry** with `temperature=0`, `num_predict=1024`.
- **Straggler mode** (`/api/index/vlm/stragglers`) — simplified prompt with
  only `text` + `caption` and `num_predict=3072`. For dense study screenshots
  and complex layouts that overflow the regular prompt's token budget.
- **Sentinel** caption `[авто: не удалось распознать]` for genuinely
  unrescuable memes (4-6 out of 4000), so they fall out of future queues.

## API

| Endpoint | Description |
|---|---|
| `GET /api/stats` | Stats + progress for every background job |
| `GET /api/search?q=...&mode=ocr\|clip&limit=&offset=` | Legacy single-retriever |
| `POST /api/search` (multipart) | **Phase 3 hybrid** — `query`, `mode`, `image`, `limit`, `offset` |
| `GET /api/image/original/{filename}` | Original image from `memes/` |
| `POST /api/index/start` | Full folder indexing |
| `POST /api/index/vlm` | VLM backfill for memes without a caption |
| `POST /api/index/vlm/stragglers` | Compact-prompt rescue for hard memes |
| `POST /api/index/text-embeddings` | Backfill BGE-M3 vectors |
| `POST /api/index/reocr` | Re-OCR all memes (after changing `OCR_ENGINE` / `PREPROCESS`) |
| `POST /api/index/resync` | Clean up drift between SQLite and ChromaDB |

## Configuration

Everything in `backend/config.py`:

```python
# Models
CLIP_TEXT_MODEL_NAME = "clip-ViT-B-32-multilingual-v1"
CLIP_IMAGE_MODEL_NAME = "clip-ViT-B-32"
TEXT_EMBED_MODEL_NAME = "BAAI/bge-m3"
VLM_MODEL = "huihui_ai/qwen3-vl-abliterated:8b-instruct"

# OCR
OCR_LANGUAGES = ["ru", "en"]
OCR_ENGINE = "ensemble"   # "easyocr" | "paddle" | "ensemble"
PREPROCESS = True         # CLAHE / unsharp / bilateral / adaptive threshold

# Hybrid search (Phase 3)
SEARCH_USE_FTS = True
SEARCH_USE_TEXT_VECTOR = True
SEARCH_USE_IMAGE_VECTOR = True
RRF_K1 = 60.0
RRF_K2 = 20.0
RETRIEVER_TOP_K = 60
IMAGE_RETRIEVER_TOP_K = 40

# Misc
BATCH_SIZE = 16
INDEX_LIMIT = 0           # 0 = no limit
DEVICE = "cuda" if available, else "cpu"
```

## Migrating from older versions (Phase 1/2 -> Phase 3)

If you already have an indexed database on the older schema:

```bash
cd backend
python scripts/migrate_vlm_fields.py
```

The script is idempotent — it adds the `text_for_embedding` column (if missing),
recomputes it for every row, and uploads BGE vectors into the new ChromaDB
collection. Alternative: start the backend and call
`POST /api/index/text-embeddings`.

## Tests

```bash
cd backend
pytest                        # all (including slow CLIP tests)
pytest -m "not slow"          # 55 tests, no model downloads
pytest tests/test_search.py   # Phase 3 only
```

## Privacy

`.gitignore` covers `memes/`, `memefinder.db*`, `chroma_data/`, `*.log`, `.env`.
The database stores base64 thumbnails — **never** commit it. The repo only
contains `memes/meme1.jpg` and `memes/meme2.jpg` for smoke tests.

## Known limitations

- The 8B abliterated VLM occasionally (~0.2%) gets stuck in a repetition loop
  on stylized fonts or hand-drawn schemas — those memes get the sentinel caption.
- BGE-M3 is the heaviest model (~2.3 GB); on CPU each query is ~50 ms, on CUDA
  it drops to ~5 ms.
- A first full index of a large library (3000+ memes) with VLM enabled takes
  20-30 hours on the 8B model. Subsequent additions are seconds per image.
