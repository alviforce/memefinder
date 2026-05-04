# MemeFinder

Локальная поисковая система по мемам. Индексирует папку `memes/` и ищет нужный
мем по тексту, по смыслу или по картинке — всё локально, без облаков.

Поиск гибридный: один запрос проходит через три ретривера одновременно
(SQLite FTS5 keyword, BGE-M3 семантика, CLIP визуал) и результаты сливаются
через **Reciprocal Rank Fusion** в одну ранжированную выдачу.

## Возможности

- 🌟 **Гибридный поиск** — FTS5 + BGE-M3 + CLIP, объединённые через RRF.
- 📝 **OCR** через EasyOCR + PaddleOCR в ансамбле, с CLAHE-препроцессингом
  под шакальные мемные шрифты.
- 🤖 **VLM-описания** через Ollama (`qwen3-vl-abliterated:8b-instruct`) —
  caption, объяснение шутки, теги.
- 🧠 **CLIP** для поиска по смыслу и по загруженной картинке.
- 📚 **BGE-M3** (multilingual, dim=1024) для семантики над OCR + VLM-полями.
- ⚡ **Infinite scroll** на фронте, async-индексация на бэке.
- 🔒 **Полностью локально** — никаких внешних API, всё на твоём железе.

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

При первом запуске скачаются ML-модели (~3 GB суммарно):
- `clip-ViT-B-32-multilingual-v1` (~500 MB)
- `clip-ViT-B-32` (~340 MB)
- `BAAI/bge-m3` (~2.3 GB)
- EasyOCR в `~/.EasyOCR/`, PaddleOCR в `~/.paddleocr/`

### 2. Ollama (для VLM-описаний)

```bash
ollama pull huihui_ai/qwen3-vl-abliterated:8b-instruct
ollama serve
```

Если Ollama не запущен — индексация пройдёт без VLM-полей (только OCR + CLIP).

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Фронт по адресу `http://localhost:5173`. Vite проксирует `/api` → `http://127.0.0.1:8000`.

### 4. Загрузить мемы и проиндексировать

Закинь картинки (`.jpg` / `.jpeg` / `.png` / `.webp`) в папку `memes/`, открой
фронт, нажми **«🚀 Начать индексацию из папки memes/»**. Можно следить за
прогрессом — он стримится через `/api/stats`.

## Использование поиска

В верхней панели три режима:

| Режим | Что делает | Когда выбирать |
|---|---|---|
| 🌟 **Гибридный** | FTS5 + BGE-M3 + CLIP, RRF-fusion | По умолчанию — лучший результат |
| 📝 **По тексту** | FTS5 + BGE-M3 (без CLIP) | Когда уверен в формулировке |
| 🧠 **По смыслу** | Только CLIP | Когда нужен чисто визуал |

📷-кнопка справа от строки поиска — загрузить картинку для image-image поиска.
Можно комбинировать с текстом: "найди мемы похожие на эту картинку и про котов".

## Архитектура

### Хранилища

Каждый мем живёт в двух местах, синхронизация через `/api/index/resync`:

- **SQLite** (`memefinder.db`) — `memes` (filename, ocr_text, thumbnail_base64,
  caption, humor_explain, tags, text_for_embedding) + `memes_fts` (FTS5 BM25
  по `ocr_text`). Thread-local connections, WAL mode.
- **ChromaDB** (`chroma_data/`) — две коллекции:
  - `meme_embeddings` — CLIP image embeddings (dim=512, cosine).
  - `text_embeddings` — BGE-M3 dense vectors (dim=1024, cosine) над
    `text_for_embedding = OCR + caption + humor + tags`.

### Pipeline индексации

Все CPU/GPU-операции — в `ThreadPoolExecutor(max_workers=1)`, чтобы не блокировать
async-цикл и не ловить race-condition'ы на ML-моделях.

Для каждого файла определяется состояние:
1. В SQLite + обоих коллекциях ChromaDB → **skip**.
2. В SQLite, не хватает CLIP/BGE-вектора → **embed-only** (быстрый путь).
3. Новый → **full pipeline**:
   - OCR (EasyOCR + PaddleOCR ансамбль с CV-препроцессингом)
   - VLM (Ollama → JSON с `text`, `caption`, `humor`, `tags`)
   - thumbnail (PIL → base64)
   - CLIP image embedding
   - BGE-M3 text embedding
   - persist в SQLite + обе коллекции ChromaDB

### Гибридный поиск

`backend/search_service.py:hybrid_search()`:

1. Параллельно (через executor) запускаются ретриверы:
   - **FTS5** — top-60 по BM25 над `ocr_text`.
   - **BGE-M3** — энкодим запрос → top-60 cosine-similarity в `text_embeddings`.
   - **CLIP** — text→image (если только текст) или image→image (если загружена картинка) — top-40 в `meme_embeddings`.
2. **Reciprocal Rank Fusion** объединяет ранги:
   `score = Σ weight_l / (k1 + rank_l × k2)`, k1=60, k2=20.
   Веса: BGE — 1.2 (самый сильный сигнал на ru/en), CLIP-text — 0.7, остальные — 1.0.
3. Дедуп → пагинация → batch-fetch метаданных из SQLite.
4. В ответе есть `timing_ms` каждого этапа — удобно тюнить.

### VLM-описания

Используется `qwen3-vl-abliterated:8b-instruct` через Ollama с `format="json"` для
строгого JSON-ответа `{text, caption, humor, tags}`. На случай обрезки JSON или
`temperature`-капризов модели:
- **Один retry** с `temperature=0`, `num_predict=1024`.
- **«Straggler-режим»** (`/api/index/vlm/stragglers`) — упрощённый prompt только
  для `text` + `caption` с `num_predict=3072`. Для конспектов и сложных
  скриншотов где модель не успевала уложиться в обычный prompt.
- **Sentinel** `[авто: не удалось распознать]` для безнадёжных мемов
  (4-6 штук на 4000), чтобы не пытаться снова при каждом прогоне.

## API

| Endpoint | Описание |
|---|---|
| `GET /api/stats` | Статистика + прогресс всех фоновых задач |
| `GET /api/search?q=...&mode=ocr\|clip&limit=&offset=` | Legacy single-retriever |
| `POST /api/search` (multipart) | **Phase 3 hybrid** — `query`, `mode`, `image`, `limit`, `offset` |
| `GET /api/image/original/{filename}` | Оригинал картинки из `memes/` |
| `POST /api/index/start` | Полная индексация папки |
| `POST /api/index/vlm` | VLM-backfill для мемов без caption |
| `POST /api/index/vlm/stragglers` | Компактный prompt для «трудных» мемов |
| `POST /api/index/text-embeddings` | Backfill BGE-M3 векторов |
| `POST /api/index/reocr` | Re-OCR всех мемов (после смены `OCR_ENGINE` / `PREPROCESS`) |
| `POST /api/index/resync` | Чистка drift между SQLite и ChromaDB |

## Конфигурация

Всё в `backend/config.py`:

```python
# Модели
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

# Прочее
BATCH_SIZE = 16
INDEX_LIMIT = 0           # 0 = no limit
DEVICE = "cuda" если CUDA доступна, иначе "cpu"
```

## Миграция со старой версии (Phase 1/2 → Phase 3)

Если у тебя уже есть проиндексированная база со старой схемой:

```bash
cd backend
python scripts/migrate_vlm_fields.py
```

Скрипт идемпотентный — добавляет колонку `text_for_embedding` (если нужно),
вычисляет её для всех строк и заливает BGE-векторы во вторую Chroma-коллекцию.
Альтернатива — запустить бэкенд и дёрнуть `POST /api/index/text-embeddings`.

## Тесты

```bash
cd backend
pytest                        # всё (включая slow CLIP-тесты)
pytest -m "not slow"          # без скачивания моделей — 55 тестов
pytest tests/test_search.py   # только Phase 3
```

## Приватность

`.gitignore` закрывает `memes/`, `memefinder.db*`, `chroma_data/`, `*.log`, `.env`.
БД хранит base64-миниатюры — **никогда** не коммить её. В git'е есть только
тестовые `memes/meme1.jpg` и `memes/meme2.jpg` для smoke-тестов.

## Известные ограничения

- VLM (8B abliterated) иногда (≈0.2%) залипает на repetition loop'ах с шакальными
  шрифтами или схемами от руки — для них проставляется sentinel-метка.
- BGE-M3 — самая тяжёлая модель (~2.3 GB), на CPU работает ~50 мс/запрос.
  С CUDA ~5 мс.
- Первая индексация большой коллекции (3000+ мемов) с VLM — 20-30 часов
  на 8B-модели. Дальше при добавлении новых картинок — секунды на штуку.
