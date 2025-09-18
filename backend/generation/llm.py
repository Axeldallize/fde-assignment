from __future__ import annotations

from typing import Dict, Any

from backend.config import settings


def generate_answer(prompt: str, temperature: float = 0.1) -> str:
    """Call Anthropic Claude with the given prompt and return text.

    Note: Minimal wrapper; proper tool use/JSON modes can be added later.
    """
    if settings.llm_provider != "anthropic":
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    try:
        from anthropic import Anthropic
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("anthropic package not installed. Add anthropic to requirements.") from exc

    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    client = Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=800,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    # anthropic SDK returns content as a list of blocks
    parts = getattr(resp, "content", [])
    if not parts:
        return ""
    # text blocks contain a 'text' field
    return "\n".join([getattr(p, "text", "") for p in parts if getattr(p, "type", "") == "text"]) or ""


