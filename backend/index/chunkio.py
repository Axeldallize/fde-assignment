from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

from .store import chunks_dir


def _doc_id_from_chunk_id(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[0]


def load_id_to_text_for_doc(doc_id: str) -> Dict[str, str]:
    cdir = chunks_dir()
    texts_path = cdir / f"{doc_id}.texts.json"
    map_path = cdir / f"{doc_id}.map.json"
    texts = json.loads(texts_path.read_text(encoding="utf-8"))
    id_map = json.loads(map_path.read_text(encoding="utf-8"))
    # id_map: chunk_id -> index in texts
    return {cid: texts[idx] for cid, idx in id_map.items()}


def load_id_to_meta_for_doc(doc_id: str) -> Dict[str, Dict]:
    cdir = chunks_dir()
    jsonl_path = cdir / f"{doc_id}.jsonl"
    out: Dict[str, Dict] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[obj["chunk_id"]] = obj
    return out


def get_text_map_for_ids(chunk_ids: List[str]) -> Dict[str, str]:
    # group by doc_id and merge per-doc maps
    out: Dict[str, str] = {}
    by_doc: Dict[str, List[str]] = {}
    for cid in chunk_ids:
        by_doc.setdefault(_doc_id_from_chunk_id(cid), []).append(cid)
    for doc_id, ids in by_doc.items():
        id2text = load_id_to_text_for_doc(doc_id)
        for cid in ids:
            if cid in id2text:
                out[cid] = id2text[cid]
    return out


def get_meta_map_for_ids(chunk_ids: List[str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    by_doc: Dict[str, List[str]] = {}
    for cid in chunk_ids:
        by_doc.setdefault(_doc_id_from_chunk_id(cid), []).append(cid)
    for doc_id, ids in by_doc.items():
        id2meta = load_id_to_meta_for_doc(doc_id)
        for cid in ids:
            if cid in id2meta:
                out[cid] = id2meta[cid]
    return out


