"""
MemeFinder — Configuration
"""
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ── Пути ─────────────────────────────────────────────────────────────────────
MEMES_DIR = ROOT_DIR / "memes"
MEMES_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(__file__).resolve().parent / "memefinder.db"
CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_data")

# ── CLIP ──────────────────────────────────────────────────────────────────────
# Text: multilingual model (ru/en/…) aligned to CLIP ViT-B/32 embedding space
CLIP_TEXT_MODEL_NAME = "clip-ViT-B-32-multilingual-v1"
# Images: original CLIP ViT-B/32 — same embedding space as the multilingual text model
CLIP_IMAGE_MODEL_NAME = "clip-ViT-B-32"

# ── Text embeddings (BGE-M3 — Phase 3 hybrid search) ─────────────────────────
# Multilingual dense text embedder (ru/en, dim=1024) — used for semantic search
# over OCR + VLM caption + humor + tags. Lives in ChromaDB collection
# `text_embeddings` next to the existing CLIP image collection.
TEXT_EMBED_MODEL_NAME = "BAAI/bge-m3"

# ── OCR ──────────────────────────────────────────────────────────────────────
OCR_LANGUAGES = ["ru", "en"]

# Which OCR engine to use:
#   "easyocr"  — original EasyOCR only
#   "paddle"   — PaddleOCR only (better on stylized meme fonts, ru/en)
#   "ensemble" — run both and merge results (best recall, slowest)
OCR_ENGINE = "ensemble"

# Apply CLAHE / unsharp mask / bilateral denoise / adaptive threshold
# to images before OCR. Improves recall on gradient/shadow/stylized memes.
PREPROCESS = True

# ── VLM (Vision-Language Model via Ollama) ───────────────────────────────────
VLM_ENABLED = True
# Phase 3: switched to the Qwen3-VL instruct model (better at meme captions)
VLM_MODEL = "huihui_ai/qwen3-vl-abliterated:8b-instruct"
OLLAMA_HOST = "http://localhost:11434"
VLM_TIMEOUT = 180  # seconds per image — VLM is slow

# ── Батчинг ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
INDEX_LIMIT = 0  # 0 = no limit

# ── Миниатюры ────────────────────────────────────────────────────────────────
THUMBNAIL_SIZE = (150, 150)
THUMBNAIL_QUALITY = 60

# ── API ──────────────────────────────────────────────────────────────────────
DEFAULT_SEARCH_LIMIT = 30

# ── Hybrid search (Phase 3) ──────────────────────────────────────────────────
# Toggle individual retrievers for the hybrid endpoint. Useful for benchmarking.
SEARCH_USE_FTS = True            # SQLite FTS5 keyword search over OCR text
SEARCH_USE_TEXT_VECTOR = True    # BGE-M3 dense semantic search
SEARCH_USE_IMAGE_VECTOR = True   # CLIP image-space search (text→image and image→image)

# Reciprocal Rank Fusion constants. Higher k1 dampens top-rank dominance,
# higher k2 makes the rank-decay steeper. See search_service.reciprocal_rank_fusion.
RRF_K1 = 60.0
RRF_K2 = 20.0

# Per-retriever candidate pool size before fusion. Bigger = more recall, slower.
RETRIEVER_TOP_K = 60
IMAGE_RETRIEVER_TOP_K = 40

# ── Устройство ───────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
