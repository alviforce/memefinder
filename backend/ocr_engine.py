"""
MemeFinder v2.0 — EasyOCR wrapper
Extracts text from meme images in-memory (bytes, no disk).
Supports batch processing with VRAM cleanup.
"""
import gc
import logging
import numpy as np
from PIL import Image
import io

from config import OCR_LANGUAGES, DEVICE

logger = logging.getLogger(__name__)

_reader = None


def _get_reader():
    """Lazy-init EasyOCR reader (downloads models on first use)."""
    global _reader
    if _reader is None:
        import easyocr
        logger.info("Initializing EasyOCR reader for languages: %s", OCR_LANGUAGES)
        _reader = easyocr.Reader(OCR_LANGUAGES, gpu=(DEVICE == "cuda"))
        logger.info("EasyOCR reader ready.")
    return _reader


def load_model():
    """Pre-load the OCR model (call at startup)."""
    _get_reader()
    logger.info("EasyOCR model loaded successfully.")


def extract_text(image_bytes: bytes) -> str:
    """
    Extract text from image bytes using EasyOCR.
    Returns a single string with all detected text joined by spaces.
    Returns empty string on failure.
    """
    try:
        reader = _get_reader()
        # EasyOCR can accept numpy arrays directly
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        results = reader.readtext(image_np, detail=0)
        text = " ".join(results).strip()
        return text
    except Exception as e:
        logger.warning("OCR failed: %s", e)
        return ""


def extract_texts_batch(images_bytes: list[bytes]) -> list[str]:
    """
    Extract text from a batch of image bytes.
    Returns list of extracted text strings.
    Performs VRAM cleanup after processing.
    """
    results = []
    try:
        import torch
        reader = _get_reader()

        for img_bytes in images_bytes:
            try:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image_np = np.array(image)
                ocr_results = reader.readtext(image_np, detail=0)
                text = " ".join(ocr_results).strip()
                results.append(text)
            except Exception as e:
                logger.warning("OCR failed for image in batch: %s", e)
                results.append("")
    except Exception as e:
        logger.error("Batch OCR failed: %s", e)
        # Pad with empty strings for failed batch
        while len(results) < len(images_bytes):
            results.append("")
    finally:
        # VRAM cleanup
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return results
