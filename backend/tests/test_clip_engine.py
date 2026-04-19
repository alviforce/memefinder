"""
Tests for clip_engine.py — CLIP encoding.
Loads the actual model (requires sentence-transformers).
Marked slow so they can be skipped in CI: pytest -m "not slow"
"""
import math
import pytest
import numpy as np


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def loaded_clip():
    """Load CLIP model once for all tests in this module."""
    import clip_engine
    clip_engine.load_models()
    return clip_engine


def test_encode_text_returns_list(loaded_clip):
    result = loaded_clip.encode_text("кот")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)


def test_encode_text_is_normalized(loaded_clip):
    result = loaded_clip.encode_text("смешной мем")
    norm = math.sqrt(sum(x * x for x in result))
    assert abs(norm - 1.0) < 1e-5, f"Expected norm≈1, got {norm}"


def test_encode_text_dimension(loaded_clip):
    """ViT-B-32 produces 512-dimensional embeddings."""
    result = loaded_clip.encode_text("test")
    assert len(result) == 512


def test_encode_text_different_for_different_queries(loaded_clip):
    emb1 = loaded_clip.encode_text("кот")
    emb2 = loaded_clip.encode_text("автомобиль")
    dot = sum(a * b for a, b in zip(emb1, emb2))
    assert dot < 0.99, "Different queries should produce different embeddings"


def test_encode_image_returns_list(loaded_clip, sample_image):
    result = loaded_clip.encode_image(sample_image)
    assert isinstance(result, list)
    assert len(result) == 512


def test_encode_image_is_normalized(loaded_clip, sample_image):
    result = loaded_clip.encode_image(sample_image)
    norm = math.sqrt(sum(x * x for x in result))
    assert abs(norm - 1.0) < 1e-5


def test_encode_images_batch(loaded_clip, sample_image):
    from PIL import Image
    images = [
        sample_image,
        Image.new("RGB", (64, 64), color=(255, 0, 0)),
        Image.new("RGB", (64, 64), color=(0, 255, 0)),
    ]
    results = loaded_clip.encode_images_batch(images)
    assert len(results) == 3
    for emb in results:
        assert len(emb) == 512
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 1e-5


def test_encode_images_batch_empty(loaded_clip):
    assert loaded_clip.encode_images_batch([]) == []


def test_text_image_similarity(loaded_clip, sample_image):
    """Text and image embeddings should be in the same space (dot product finite)."""
    text_emb = np.array(loaded_clip.encode_text("цветной прямоугольник"))
    img_emb = np.array(loaded_clip.encode_image(sample_image))
    similarity = float(np.dot(text_emb, img_emb))
    assert -1.0 <= similarity <= 1.0
