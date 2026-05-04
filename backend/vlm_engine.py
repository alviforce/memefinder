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

# Compact prompt for straggler memes (учебные конспекты, длинные скрины поста, шакальный шрифт).
# Обрезаем требования: текста max 300 символов, без `humor` и `tags` — экономит ~50% токенов
# и сильно повышает шанс уложить ответ в num_predict.
STRAGGLER_PROMPT = (
    "Ты анализируешь сложный мем — возможно конспект, схему, длинный скриншот или плохо читаемый текст.\n"
    "Верни СТРОГИЙ JSON ровно с двумя ключами:\n"
    "- \"text\": главные слова с картинки (НЕ БОЛЕЕ 300 символов; если текста много — выбери самое заметное)\n"
    "- \"caption\": одно короткое предложение по-русски о чём картинка (НЕ БОЛЕЕ 150 символов)\n"
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


def _ollama_chat(
    image_b64: str,
    *,
    temperature: float,
    num_predict: int,
    prompt: str = PROMPT,
) -> str:
    """Single chat call. Returns the raw content string (may be empty)."""
    client = _get_client()
    resp = client.chat(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
        format="json",
        options={"temperature": temperature, "num_predict": num_predict},
    )
    if hasattr(resp, "message") and hasattr(resp.message, "content"):
        return resp.message.content or ""
    if isinstance(resp, dict):
        return (resp.get("message") or {}).get("content", "") or ""
    return str(resp or "")


def describe_meme(image_bytes: bytes, filename: str | None = None) -> dict | None:
    """
    Run VLM on a single meme. Returns dict with keys
      {"text": str, "caption": str, "humor": str, "tags": list[str]}
    or None on failure / when VLM is disabled.

    On a JSON-parse failure or empty content we retry once with temperature=0
    and num_predict=1024 — the abliterated 8B VL model occasionally either
    truncates JSON or wraps it in prose, and a stricter second pass usually
    fixes it. `filename` is purely for log context.
    """
    tag = filename or "<unknown>"
    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")

        # ── Attempt 1: fast path ─────────────────────────────────────────────
        content = _ollama_chat(b64, temperature=0.2, num_predict=1024)
        data = _extract_json(content) if content else None

        # ── Attempt 2: deterministic + bigger token budget ───────────────────
        if not isinstance(data, dict):
            preview = (content or "").strip().replace("\n", " ")[:200]
            logger.warning(
                "VLM[%s] attempt 1 failed (empty=%s, len=%d). Raw: %r — retrying "
                "with temperature=0, num_predict=1024.",
                tag, not content, len(content or ""), preview,
            )
            content2 = _ollama_chat(b64, temperature=0.0, num_predict=1024)
            data = _extract_json(content2) if content2 else None
            if not isinstance(data, dict):
                preview2 = (content2 or "").strip().replace("\n", " ")[:200]
                logger.warning(
                    "VLM[%s] retry also failed (empty=%s, len=%d). Raw: %r — giving up.",
                    tag, not content2, len(content2 or ""), preview2,
                )
                return None

        return {
            "text": str(data.get("text", "") or "").strip(),
            "caption": str(data.get("caption", "") or "").strip(),
            "humor": str(data.get("humor", "") or "").strip(),
            "tags": _normalize_tags(data.get("tags", [])),
        }
    except Exception as e:
        logger.warning("VLM[%s] crashed: %s", tag, e, exc_info=True)
        return None


class VLMServiceUnavailable(RuntimeError):
    """Raised when Ollama itself is unreachable (network / process down)."""


def _is_connection_error(exc: BaseException) -> bool:
    """True if the exception looks like 'Ollama is not responding'."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return any(token in name for token in ("connect", "timeout")) or any(
        token in msg
        for token in ("connection refused", "connection error", "failed to connect",
                      "max retries exceeded", "no connection", "actively refused")
    )


def describe_meme_compact(image_bytes: bytes, filename: str | None = None) -> dict | None:
    """
    Compact prompt mode for «straggler» memes that the standard prompt failed to
    caption — typically dense study material, long screenshots, or stylized fonts
    that blow past `num_predict`. We:
      • drop the humor + tags fields (saves ~50 % tokens)
      • cap text/caption length in the prompt itself
      • give the model 3072 tokens of head-room
      • do a single deterministic pass (temperature=0)

    Returns the same dict shape as `describe_meme` so callers can stay generic;
    `humor` and `tags` come back empty/[] for these memes.

    Raises `VLMServiceUnavailable` when Ollama itself is unreachable so the
    caller can distinguish a model failure (=> sentinel-mark the meme) from a
    service outage (=> retry later, do NOT mark anything).
    """
    tag = filename or "<unknown>"
    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content = _ollama_chat(
            b64, temperature=0.0, num_predict=3072, prompt=STRAGGLER_PROMPT,
        )
        data = _extract_json(content) if content else None
        if not isinstance(data, dict):
            preview = (content or "").strip().replace("\n", " ")[:200]
            logger.warning(
                "VLM-compact[%s] failed (empty=%s, len=%d). Raw: %r",
                tag, not content, len(content or ""), preview,
            )
            return None
        return {
            "text": str(data.get("text", "") or "").strip(),
            "caption": str(data.get("caption", "") or "").strip(),
            "humor": "",
            "tags": [],
        }
    except Exception as e:
        if _is_connection_error(e):
            # Don't bury connection errors — caller decides what to do.
            raise VLMServiceUnavailable(str(e)) from e
        logger.warning("VLM-compact[%s] crashed: %s", tag, e, exc_info=True)
        return None
