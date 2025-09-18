from __future__ import annotations

import re


_RE_WS = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _RE_WS.sub(" ", text).strip()


def count_tokens(text: str) -> int:
    # Rough token proxy: whitespace word count
    if not text:
        return 0
    return len(text.split())


def tail_words(text: str, max_words: int) -> str:
    if max_words <= 0 or not text:
        return ""
    words = text.split()
    return " ".join(words[-max_words:])


