from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import json

import numpy as np

from backend.config import settings
from .store import index_dir, chunks_dir, write_json, read_json


_EMB_MATRIX_PATH = index_dir() / "embeddings.npy"
_EMB_IDS_PATH = index_dir() / "embedding_ids.json"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray: # Note: I use cosine similarity for semantic similarity instead of dot product because it is more stable and easier to compute.
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def _embed_voyage(texts: List[str], model: str, batch_size: int = 128) -> np.ndarray:
    try:
        import voyageai  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("voyageai package not installed. Add voyageai to requirements.") from exc

    if not settings.voyage_api_key:
        # Soft-fail to avoid 500s; return zeros so semantic contributes nothing
        return np.zeros((len(texts), 384), dtype=np.float32)

    client = voyageai.Client(api_key=settings.voyage_api_key)
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embed(batch, model=model)
        embeddings.extend(resp.embeddings)
    return np.asarray(embeddings, dtype=np.float32)


def save_embeddings(matrix: np.ndarray, ids: List[str]) -> Dict[str, Path]:
    out_dir = index_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(_EMB_MATRIX_PATH, matrix)
    write_json(_EMB_IDS_PATH, ids)
    return {"matrix": _EMB_MATRIX_PATH, "ids": _EMB_IDS_PATH}


def load_embeddings() -> Tuple[np.ndarray, List[str]]:
    matrix: np.ndarray = np.load(_EMB_MATRIX_PATH)
    ids: List[str] = read_json(_EMB_IDS_PATH, default=[])
    return matrix, ids


def build_embeddings_from_all_chunks(model: str | None = None) -> Dict[str, Path]: 
    """Embed all chunk texts from chunks_dir and persist a single matrix + ids.

    Returns saved paths dict.
    """
    cdir = chunks_dir()
    texts_files = sorted(cdir.glob("*.texts.json"))
    if not texts_files:
        raise ValueError("No chunk texts found. Ingest documents first.")

    corpus_texts: List[str] = []
    corpus_ids: List[str] = []
    for texts_path in texts_files:
        stem = texts_path.name.replace(".texts.json", "")
        map_path = cdir / f"{stem}.map.json"
        if not map_path.exists():
            continue
        texts = json.loads(texts_path.read_text(encoding="utf-8"))
        id_map = json.loads(map_path.read_text(encoding="utf-8"))
        ids = sorted(id_map, key=lambda k: id_map[k])
        corpus_texts.extend(texts)
        corpus_ids.extend(ids)

    if not corpus_texts:
        raise ValueError("No chunk texts to embed.")

    provider = settings.embedding_provider
    model_name = model or settings.embedding_model
    if provider != "voyage":
        # No-op build; create empty embeddings matching corpus size
        matrix = np.zeros((len(corpus_texts), 384), dtype=np.float32)
    else:
        matrix = _embed_voyage(corpus_texts, model=model_name)
    return save_embeddings(matrix, corpus_ids)


def semantic_search(query: str, top_k: int = 5, model: str | None = None) -> List[Tuple[str, float]]: # top_k is set to 4 as a reasonable compromise and can be adjusted in .env if needed.
    """Compute embedding for query using configured provider and return top_k (id, score)."""
    provider = settings.embedding_provider
    model_name = model or settings.embedding_model
    matrix, ids = load_embeddings()
    if provider != "voyage" or not settings.voyage_api_key:
        # Fallback to zeros so semantic path is neutral
        sims = np.zeros((matrix.shape[0],), dtype=np.float32)
    else:
        q_vec = _embed_voyage([query], model=model_name)
        sims = _cosine_similarity(matrix, q_vec)[..., 0]
    top_k = max(1, top_k)
    top_idx = np.argsort(-sims)[:top_k]
    return [(ids[i], float(sims[i])) for i in top_idx]


