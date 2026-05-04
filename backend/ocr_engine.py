"""
MemeFinder — OCR engine (EasyOCR + PaddleOCR + optional preprocessing).

Selects engine at runtime based on config.OCR_ENGINE:
  "easyocr"  — EasyOCR only (default legacy behavior)
  "paddle"   — PaddleOCR only
  "ensemble" — union of both (best recall on stylized meme fonts)

If config.PREPROCESS is True, every image is run through preprocess.py
to generate contrast-boosted + adaptive-threshold variants; OCR is run
over each variant and the union of detected words is returned.
"""
import gc
import io
import logging

import numpy as np
from PIL import Image

from config import OCR_LANGUAGES, DEVICE, OCR_ENGINE, PREPROCESS
import preprocess

logger = logging.getLogger(__name__)

_easyocr_reader = None
_paddle_readers: dict[str, object] = {}  # keyed by language code


# ─────────────────────────── Engine loaders ────────────────────────────────

def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        logger.info("Initializing EasyOCR for languages: %s", OCR_LANGUAGES)
        _easyocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=(DEVICE == "cuda"))
        logger.info("EasyOCR ready.")
    return _easyocr_reader


# PaddleOCR only supports one language per reader instance, so we keep one per lang.
_PADDLE_LANG_MAP = {"ru": "ru", "en": "en"}


def _get_paddle(lang: str):
    if lang in _paddle_readers:
        return _paddle_readers[lang]
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.error("paddleocr is not installed. pip install paddleocr paddlepaddle")
        raise

    logger.info("Initializing PaddleOCR for language: %s", lang)
    paddle_lang = _PADDLE_LANG_MAP.get(lang, lang)
    reader = PaddleOCR(
        use_angle_cls=True,
        lang=paddle_lang,
        use_gpu=(DEVICE == "cuda"),
        show_log=False,
    )
    _paddle_readers[lang] = reader
    logger.info("PaddleOCR (%s) ready.", lang)
    return reader


def load_model():
    """Pre-load configured OCR engines at startup."""
    engine = OCR_ENGINE.lower()
    if engine in ("easyocr", "ensemble"):
        _get_easyocr()
    if engine in ("paddle", "ensemble"):
        try:
            for lang in OCR_LANGUAGES:
                _get_paddle(lang)
        except Exception as e:
            logger.warning("PaddleOCR load failed: %s — falling back to EasyOCR only.", e)
            # Make sure we have at least something
            _get_easyocr()
    logger.info("OCR engine loaded: %s", engine)


# ─────────────────────────── Per-engine readers ────────────────────────────

def _easyocr_read(img_np: np.ndarray) -> list[str]:
    try:
        reader = _get_easyocr()
        return reader.readtext(img_np, detail=0) or []
    except Exception as e:
        logger.debug("EasyOCR read failed: %s", e)
        return []


def _paddle_read(img_np: np.ndarray) -> list[str]:
    out: list[str] = []
    for lang in OCR_LANGUAGES:
        try:
            reader = _get_paddle(lang)
            # PaddleOCR returns [[ [box, (text, conf)], ... ]]
            result = reader.ocr(img_np, cls=True)
            if not result:
                continue
            for page in result:
                if not page:
                    continue
                for line in page:
                    if not line or len(line) < 2:
                        continue
                    text_part = line[1]
                    if isinstance(text_part, (list, tuple)) and text_part:
                        text = text_part[0]
                        if text:
                            out.append(str(text))
        except Exception as e:
            logger.debug("PaddleOCR (%s) read failed: %s", lang, e)
    return out


def _run_engines(img_np: np.ndarray, engine: str) -> list[str]:
    parts: list[str] = []
    if engine in ("easyocr", "ensemble"):
        parts.extend(_easyocr_read(img_np))
    if engine in ("paddle", "ensemble"):
        parts.extend(_paddle_read(img_np))
    return parts


# ─────────────────────────── Text merging ──────────────────────────────────

def _merge_texts(chunks: list[str]) -> str:
    """Dedupe whole-chunk repeats while preserving order."""
    seen = set()
    out = []
    for chunk in chunks:
        t = (chunk or "").strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return " ".join(out)


# ─────────────────────────── Public API ────────────────────────────────────

def extract_text(image_bytes: bytes) -> str:
    """
    Extract text from a single image. Applies preprocessing (if enabled) and
    runs the configured OCR engine(s). Returns empty string on failure.
    """
    engine = OCR_ENGINE.lower()
    try:
        if PREPROCESS:
            variants = preprocess.preprocess_for_ocr(image_bytes)
            if not variants:
                return ""
        else:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            variants = [np.array(image)]

        collected: list[str] = []
        for variant in variants:
            collected.extend(_run_engines(variant, engine))
        return _merge_texts(collected)
    except Exception as e:
        logger.warning("OCR failed: %s", e)
        return ""


def extract_texts_batch(images_bytes: list[bytes]) -> list[str]:
    """
    Extract text from a batch of images. Returns a list of strings, one per
    input image (empty string on failure). Performs GPU cleanup at the end.
    """
    results: list[str] = []
    try:
        for img_bytes in images_bytes:
            try:
                results.append(extract_text(img_bytes))
            except Exception as e:
                logger.warning("OCR failed for image in batch: %s", e)
                results.append("")
    except Exception as e:
        logger.error("Batch OCR failed: %s", e)
        while len(results) < len(images_bytes):
            results.append("")
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    return results
