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

# ── EasyOCR ──────────────────────────────────────────────────────────────────
OCR_LANGUAGES = ["ru", "en"]

# ── Батчинг ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
INDEX_LIMIT = 0  # 0 = no limit

# ── Миниатюры ────────────────────────────────────────────────────────────────
THUMBNAIL_SIZE = (150, 150)
THUMBNAIL_QUALITY = 60

# ── API ──────────────────────────────────────────────────────────────────────
DEFAULT_SEARCH_LIMIT = 30

# ── Устройство ───────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
