from __future__ import annotations

from typing import List, Tuple
import numpy as np

from backend.index.semantic import _cosine_similarity
from backend.index.semantic import _embed_voyage  # reuse provider stub
from backend.config import settings


def split_sentences(text: str) -> List[str]:
    import re
    s = re.split(r"(?<=[.!?])\s+", text.strip()) if text else []
    return [t.strip() for t in s if t.strip()]


def evidence_filter(answer: str, supporting_texts: List[str], threshold: float = 0.20) -> str:
    if not answer:
        return answer
    sents = split_sentences(answer)
    if not sents or not supporting_texts:
        return answer

    # embed all sentences and supporting texts using the same embedding provider
    model = settings.embedding_model
    ctx_matrix = _embed_voyage(supporting_texts, model=model)
    sent_matrix = _embed_voyage(sents, model=model)

    sims = _cosine_similarity(sent_matrix, ctx_matrix)  # shape: [num_sents, num_ctx]
    keep: List[str] = []
    for i, sent in enumerate(sents):
        max_sim = float(sims[i].max()) if sims.shape[1] > 0 else 0.0
        if max_sim >= threshold:
            keep.append(sent)
    return " ".join(keep)


