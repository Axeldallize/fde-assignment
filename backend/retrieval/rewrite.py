from __future__ import annotations

import re


_RE_WS = re.compile(r"\s+")


def deterministic_rewrite(query: str) -> str:
    if not query:
        return ""
    q = query.strip()
    q = _RE_WS.sub(" ", q)
    # Lowercase as a simple normalizer for lexical match; keep as is for LLMs later
    q = q.lower()
    return q


