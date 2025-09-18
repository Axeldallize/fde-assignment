from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple
import json

from backend.utils.text import normalize_whitespace, count_tokens, tail_words
from backend.index.store import chunks_dir, write_json
from backend.ingestion.extract import PageContent


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    headings_path: List[str]


def _split_into_chunks_by_heading(pages: List[PageContent]) -> List[Tuple[List[int], str, List[str]]]:
    groups: List[Tuple[List[int], str, List[str]]] = []
    current_pages: List[int] = []
    current_heading: List[str] = []
    current_text: List[str] = []

    for p in pages:
        # Update heading stack with first candidate if present
        if p.heading_candidates:
            current_heading = p.heading_candidates[:1]
        current_pages.append(p.page_index)
        current_text.append(p.text)

    if current_pages:
        groups.append((current_pages, "\n".join(current_text), current_heading))
    return groups


def _window_overlaps(text: str, target_tokens: int, overlap_tokens: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + target_tokens)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def build_chunks(doc_id: str, pages: List[PageContent], target_tokens: int = 1000, overlap_ratio: float = 0.15) -> List[Chunk]:
    # heading-bounded coarse grouping, then token windows with overlap
    groups = _split_into_chunks_by_heading(pages)
    overlap_tokens = max(1, int(target_tokens * overlap_ratio))
    out: List[Chunk] = []
    ordinal = 0
    for page_ids, text, heading_path in groups:
        text_norm = normalize_whitespace(text)
        windows = _window_overlaps(text_norm, target_tokens, overlap_tokens)
        if not windows:
            continue
        for w in windows:
            ordinal += 1
            out.append(
                Chunk(
                    chunk_id=f"{doc_id}::ch{ordinal}",
                    doc_id=doc_id,
                    text=w,
                    page_start=page_ids[0],
                    page_end=page_ids[-1],
                    headings_path=heading_path,
                )
            )
    return out


def persist_chunks(doc_id: str, chunks: List[Chunk]) -> Dict[str, Path]:
    # Write JSONL and a sidecar mapping for quick indexing
    out_dir = chunks_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{doc_id}.jsonl"
    texts_path = out_dir / f"{doc_id}.texts.json"
    map_path = out_dir / f"{doc_id}.map.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

    texts = [c.text for c in chunks]
    id_map = {c.chunk_id: i for i, c in enumerate(chunks)}
    write_json(texts_path, texts)
    write_json(map_path, id_map)
    return {"jsonl": jsonl_path, "texts": texts_path, "map": map_path}


