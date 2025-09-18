from __future__ import annotations

from typing import Dict, List, Tuple


def _to_score_map(results: List[Tuple[str, float]], min_norm: float = 1e-9) -> Dict[str, float]:
    if not results:
        return {}
    scores = [s for _, s in results]
    mx = max(scores) if scores else 1.0
    if mx < min_norm:
        mx = min_norm
    return {cid: s / mx for cid, s in results}


def weighted_sum(
    lexical: List[Tuple[str, float]],
    semantic: List[Tuple[str, float]],
    w_lex: float = 0.6,
    w_sem: float = 0.4,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    ls = _to_score_map(lexical)
    ss = _to_score_map(semantic)
    keys = set(ls) | set(ss)
    out = []
    for k in keys:
        out.append((k, w_lex * ls.get(k, 0.0) + w_sem * ss.get(k, 0.0)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]


def rrf(
    lexical: List[Tuple[str, float]],
    semantic: List[Tuple[str, float]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    ranks: Dict[str, float] = {}
    for lst in [lexical, semantic]:
        for rank, (cid, _score) in enumerate(lst, start=1):
            ranks[cid] = ranks.get(cid, 0.0) + 1.0 / (k + rank)
    out = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return out[:top_k]


