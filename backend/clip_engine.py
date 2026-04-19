"""
MemeFinder — CLIP engine (sentence-transformers)

Two-model design:
  - Text queries  → clip-ViT-B-32-multilingual-v1  (ru/en/… multilingual)
  - Image batches → clip-ViT-B-32                  (original CLIP, same embedding space)

The multilingual text model was trained via knowledge-distillation from
clip-ViT-B-32, so text and image embeddings are in the same space and
cosine similarity between them is meaningful.
"""
import gc
import logging
from PIL import Image
from config import CLIP_TEXT_MODEL_NAME, CLIP_IMAGE_MODEL_NAME, DEVICE

logger = logging.getLogger(__name__)

_text_model = None
_image_model = None


def load_models():
    """Load both CLIP models. Called once at startup."""
    global _text_model, _image_model
    if _text_model is not None:
        return
    from sentence_transformers import SentenceTransformer
    logger.info("Loading CLIP text model: %s on %s", CLIP_TEXT_MODEL_NAME, DEVICE)
    _text_model = SentenceTransformer(CLIP_TEXT_MODEL_NAME, device=DEVICE)
    logger.info("Loading CLIP image model: %s on %s", CLIP_IMAGE_MODEL_NAME, DEVICE)
    _image_model = SentenceTransformer(CLIP_IMAGE_MODEL_NAME, device=DEVICE)
    logger.info("CLIP models loaded.")


def _get_text_model():
    if _text_model is None:
        load_models()
    return _text_model


def _get_image_model():
    if _image_model is None:
        load_models()
    return _image_model


def encode_text(text: str) -> list[float]:
    """Encode a text query with the multilingual model. Returns normalized floats."""
    embedding = _get_text_model().encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding.tolist()


def encode_image(image: Image.Image) -> list[float]:
    """Encode a single PIL Image. Returns a normalized float list."""
    try:
        embedding = _get_image_model().encode(
            image.convert("RGB"),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
    except Exception as e:
        logger.warning("encode_image failed: %s", e)
        return []


def encode_images_batch(images: list[Image.Image]) -> list[list[float]]:
    """Encode a batch of PIL Images. Returns list of normalized float lists."""
    if not images:
        return []
    try:
        rgb = [img.convert("RGB") for img in images]
        embeddings = _get_image_model().encode(
            rgb,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=len(rgb),
            show_progress_bar=False,
        )
        return embeddings.tolist()
    except Exception as e:
        logger.error("encode_images_batch failed: %s", e)
        return []
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
