from __future__ import annotations

from typing import List, Tuple, Dict


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def rerank_by_heuristics(
    query: str,
    candidates: List[Tuple[str, float]],
    chunk_text_map: Dict[str, str],
    headings_map: Dict[str, str] | None = None,
    w_fusion: float = 0.7,
    w_coverage: float = 0.25,
    w_heading: float = 0.05,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Re-score fused candidates with simple coverage/heading heuristics.

    - query-term coverage: fraction of unique query tokens present in chunk text
    - heading match bonus: +w_heading if any query token occurs in heading
    The base fused score is combined as: w_fusion*fused + w_coverage*coverage + heading_bonus
    """
    q_tokens = set(_tokenize(query))
    out: List[Tuple[str, float]] = []
    for cid, fused_score in candidates:
        text = chunk_text_map.get(cid, "")
        if not text:
            coverage = 0.0
        else:
            t_tokens = set(_tokenize(text))
            overlap = len(q_tokens & t_tokens)
            coverage = (overlap / max(1, len(q_tokens)))
        heading_bonus = 0.0
        if headings_map:
            heading = (headings_map.get(cid) or "").lower()
            if any(tok in heading for tok in q_tokens):
                heading_bonus = w_heading
        score = w_fusion * fused_score + w_coverage * coverage + heading_bonus
        out.append((cid, score))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]


