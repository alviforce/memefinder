"""
MemeFinder — Image preprocessing for OCR.
Applies CLAHE, unsharp mask, bilateral denoising, and adaptive thresholding
to make stylized meme text more readable before OCR.

All functions accept and return numpy arrays (uint8, RGB or grayscale).
"""
import logging
import io

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Upscale small images so OCR has more pixels to work with
_MIN_SIDE = 900
# Cap very large images so preprocessing stays fast
_MAX_SIDE = 2400


def _resize_for_ocr(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    shortest = min(h, w)

    if shortest < _MIN_SIDE:
        scale = _MIN_SIDE / shortest
    elif longest > _MAX_SIDE:
        scale = _MAX_SIDE / longest
    else:
        return img

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the L channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _unsharp_mask(img: np.ndarray, amount: float = 1.2, radius: int = 3) -> np.ndarray:
    """Standard unsharp mask — subtracts a blurred copy to boost edges."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _bilateral_denoise(img: np.ndarray) -> np.ndarray:
    """Edge-preserving denoising — keeps letter strokes crisp."""
    return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)


def _adaptive_threshold(img: np.ndarray) -> np.ndarray:
    """Binarize with adaptive mean threshold. Good for gradient backgrounds."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    # Gaussian adaptive thresholding handles uneven lighting / gradients
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def preprocess_for_ocr(image_bytes: bytes, include_binary: bool = True) -> list[np.ndarray]:
    """
    Produce preprocessed variants of a meme image for OCR.

    Returns a list of numpy arrays (RGB, uint8). An OCR caller can run each
    variant and merge the text — different variants win on different memes.

    Variants:
      1. Contrast-boosted + sharpened + denoised (color).
      2. Adaptive-threshold binary (helps on gradients / dark-on-dark).

    On failure, returns [original_rgb] so the caller still has something to OCR.
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(pil)
    except Exception as e:
        logger.warning("preprocess: cannot decode image: %s", e)
        return []

    try:
        resized = _resize_for_ocr(rgb)
        clahe = _apply_clahe(resized)
        denoised = _bilateral_denoise(clahe)
        sharp = _unsharp_mask(denoised)

        variants = [sharp]
        if include_binary:
            try:
                variants.append(_adaptive_threshold(sharp))
            except Exception as e:
                logger.debug("adaptive threshold failed: %s", e)
        return variants
    except Exception as e:
        logger.warning("preprocess failed, falling back to original: %s", e)
        return [rgb]
