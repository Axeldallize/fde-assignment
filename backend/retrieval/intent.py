from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple


Intent = Literal["smalltalk", "qa", "unknown"]


_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
}

_SMALLTALK_PREFIXES = (
    "how are you",
    "what's up",
)

_QUESTION_WORDS = {
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "which",
}


@dataclass
class IntentResult:
    intent: Intent
    confidence: float


def detect_intent(query: str) -> IntentResult:
    text = (query or "").strip().lower()
    if not text:
        return IntentResult("unknown", 0.0)

    # Greetings / smalltalk
    for g in _GREETINGS:
        if text == g or text.startswith(g + " "):
            return IntentResult("smalltalk", 0.95)
    for p in _SMALLTALK_PREFIXES:
        if text.startswith(p):
            return IntentResult("smalltalk", 0.8)

    # Likely question
    if text.endswith("?"):
        return IntentResult("qa", 0.85)
    first = text.split()[0]
    if first in _QUESTION_WORDS:
        return IntentResult("qa", 0.8)

    # Longer statements default to qa intent with lower confidence
    if len(text.split()) >= 4:
        return IntentResult("qa", 0.6)

    return IntentResult("unknown", 0.3)


