"""
MemeFinder — Vision-Language Model wrapper (via Ollama).

Produces a rich description of a meme in a single pass:
  - text: exact in-image text (preserves misspellings)
  - caption: visual description (what is shown)
  - humor: explanation of the joke / cultural reference
  - tags: short list of lowercase tags (mix of ru/en)

All output is merged into SQLite fields (caption, humor_explain, tags) so
search can eventually leverage the VLM's understanding beyond raw OCR.
"""
from __future__ import annotations

import base64
import json
import logging
import re

from config import VLM_ENABLED, VLM_MODEL, OLLAMA_HOST, VLM_TIMEOUT

logger = logging.getLogger(__name__)

_client = None
_available: bool | None = None  # tri-state: None = not checked yet

PROMPT = (
    "Ты анализируешь интернет-мем. Верни СТРОГИЙ JSON со следующими ключами:\n"
    "- \"text\": полный текст на картинке (сохрани опечатки; пустая строка если текста нет)\n"
    "- \"caption\": визуальное описание мема на русском (1-2 предложения)\n"
    "- \"humor\": объяснение шутки или культурной отсылки на русском (1-2 предложения)\n"
    "- \"tags\": массив из 3-8 коротких тегов в нижнем регистре (смесь ru/en — темы, эмоции, персонажи)\n"
    "\n"
    "Ответь ТОЛЬКО JSON. Без пояснений, без markdown."
)


def _get_client():
    global _client
    if _client is None:
        import ollama
        _client = ollama.Client(host=OLLAMA_HOST, timeout=VLM_TIMEOUT)
    return _client


def is_available() -> bool:
    """Cached check whether Ollama is reachable and the VLM model is pulled."""
    global _available
    if _available is not None:
        return _available
    if not VLM_ENABLED:
        _available = False
        return False
    try:
        client = _get_client()
        models = client.list()
        # Response shape: {"models": [{"model": "name:tag", ...}, ...]} (newer)
        #              or {"models": [{"name": "name:tag", ...}, ...]}   (older)
        raw = models.get("models", []) if isinstance(models, dict) else []
        names = {(m.get("model") or m.get("name") or "") for m in raw}
        if VLM_MODEL not in names and not any(n.startswith(VLM_MODEL.split(":")[0]) for n in names):
            logger.warning(
                "VLM model %s not found in Ollama. Run: ollama pull %s",
                VLM_MODEL, VLM_MODEL,
            )
            _available = False
            return False
        _available = True
        logger.info("VLM ready: %s at %s", VLM_MODEL, OLLAMA_HOST)
        return True
    except Exception as e:
        logger.warning("Ollama not reachable (%s): VLM disabled", e)
        _available = False
        return False


def load_model() -> None:
    """Probe Ollama at startup so failures surface early."""
    is_available()


def _extract_json(text: str) -> dict | None:
    """Parse JSON from the model response, stripping code fences / stray prose."""
    if not text:
        return None
    # Strip common ```json ... ``` fences
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    # Find first {...} block if model wrote extra prose
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _normalize_tags(raw) -> list[str]:
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        items = [t for t in re.split(r"[,;\n]", raw) if t.strip()]
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for t in items:
        s = str(t).strip().lower().strip("#").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out[:12]


def describe_meme(image_bytes: bytes) -> dict | None:
    """
    Run VLM on a single meme. Returns dict with keys
      {"text": str, "caption": str, "humor": str, "tags": list[str]}
    or None on failure / when VLM is disabled.
    """
    if not is_available():
        return None
    try:
        client = _get_client()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        resp = client.chat(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": PROMPT, "images": [b64]}],
            format="json",
            options={"temperature": 0.2, "num_predict": 512},
        )
        content = (resp.get("message") or {}).get("content", "") if isinstance(resp, dict) else ""
        if not content and hasattr(resp, "message"):
            content = getattr(resp.message, "content", "")  # newer ollama-python returns objects
        data = _extract_json(content)
        if not isinstance(data, dict):
            logger.debug("VLM returned non-JSON: %r", content[:200])
            return None
        return {
            "text": str(data.get("text", "") or "").strip(),
            "caption": str(data.get("caption", "") or "").strip(),
            "humor": str(data.get("humor", "") or "").strip(),
            "tags": _normalize_tags(data.get("tags", [])),
        }
    except Exception as e:
        logger.warning("VLM failed: %s", e)
        return None
