from __future__ import annotations

from typing import List, Tuple, Dict, Any

from backend.config import settings


def mean_topk(values: List[float], k: int) -> float:
    if not values:
        return 0.0
    k = max(1, min(k, len(values)))
    top = sorted(values, reverse=True)[:k]
    return sum(top) / len(top)


def evidence_gate(
    ranked: List[Tuple[str, float]],
    chunk_doc_map: Dict[str, str],
    min_sources: int | None = None,
    threshold: float | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Decide if retrieval evidence is sufficient.

    Conditions (configurable):
    - At least min_sources distinct chunks (by doc) among top-k
    - Mean similarity of top-k >= threshold
    Returns (passed, meta)
    """
    k = settings.evidence_topk
    thr = settings.evidence_threshold if threshold is None else threshold
    need = 2 if min_sources is None else min_sources

    sims = [s for _, s in ranked]
    mt = mean_topk(sims, k)
    top_ids = [cid for cid, _ in ranked[:k]]
    docs = {chunk_doc_map.get(cid, "?") for cid in top_ids}
    passed = (len(docs) >= need) and (mt >= thr)
    meta = {"mean_topk": mt, "distinct_docs": len(docs), "need_docs": need, "k": k, "threshold": thr}
    return passed, meta


