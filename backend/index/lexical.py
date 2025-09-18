from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from .store import index_dir, write_json, read_json, chunks_dir


_VECTORIZER_PATH = index_dir() / "tfidf_vectorizer.pkl"
_MATRIX_PATH = index_dir() / "tfidf_matrix.npz"
_IDS_PATH = index_dir() / "tfidf_ids.json"


def build_index(corpus: List[Tuple[str, str]]) -> Tuple[TfidfVectorizer, sparse.csr_matrix, List[str]]: # Note: I use a TF-IDF index for lexical similarity, which is a good compromise between speed and accuracy. It has good persistence and is easy to index, at the expense of some accuracy which will be corrected by semantic similarity.
    """Build a TF-IDF index.

    corpus: list of (chunk_id, text)
    returns: (vectorizer, matrix, ids)
    """
    ids = [cid for cid, _ in corpus]
    texts = [text for _, text in corpus]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", norm="l2")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, ids


def save_index(vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix, ids: List[str]) -> Dict[str, Path]:
    out_dir = index_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, _VECTORIZER_PATH)
    sparse.save_npz(_MATRIX_PATH, matrix)
    write_json(_IDS_PATH, ids)
    return {"vectorizer": _VECTORIZER_PATH, "matrix": _MATRIX_PATH, "ids": _IDS_PATH}


def load_index() -> Tuple[TfidfVectorizer, sparse.csr_matrix, List[str]]:
    vectorizer: TfidfVectorizer = joblib.load(_VECTORIZER_PATH)
    matrix: sparse.csr_matrix = sparse.load_npz(_MATRIX_PATH)
    ids: List[str] = read_json(_IDS_PATH, default=[])
    return vectorizer, matrix, ids


def search(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    vectorizer, matrix, ids = load_index()
    q = vectorizer.transform([query])  # already l2-normalized by vectorizer
    # Note : cosine similarity = dot product since both are l2-normalized
    sims = (matrix @ q.T).toarray().ravel()
    if top_k <= 0:
        top_k = 1
    top_idx = np.argsort(-sims)[:top_k]
    return [(ids[i], float(sims[i])) for i in top_idx]


def build_index_from_all_chunks() -> Dict[str, Path]:
    """Scan chunks_dir for *.texts.json and *.map.json, rebuild and save TF-IDF index.

    Returns paths dict from save_index(). If no chunks found, raises ValueError.
    """
    import json

    cdir = chunks_dir()
    texts_files = sorted(cdir.glob("*.texts.json"))
    if not texts_files:
        raise ValueError("No chunk texts found. Ingest documents first.")

    corpus: List[Tuple[str, str]] = []
    for texts_path in texts_files:
        stem = texts_path.name.replace(".texts.json", "")
        map_path = cdir / f"{stem}.map.json"
        if not map_path.exists():
            continue
        texts = json.loads(texts_path.read_text(encoding="utf-8"))
        id_map = json.loads(map_path.read_text(encoding="utf-8"))
        ids = sorted(id_map, key=lambda k: id_map[k])
        corpus.extend(zip(ids, texts))

    if not corpus:
        raise ValueError("No chunks to index.")

    vectorizer, matrix, ids = build_index(corpus)
    return save_index(vectorizer, matrix, ids)

