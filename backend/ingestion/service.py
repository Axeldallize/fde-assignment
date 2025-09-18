from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
import shutil

from backend.index.store import ensure_data_dirs, docs_dir
from backend.ingestion.extract import extract_pdf_pages
from backend.ingestion.chunk import build_chunks, persist_chunks
from backend.ingestion.manifest import compute_md5, upsert_document
from backend.index.lexical import build_index_from_all_chunks


def ingest_files(file_paths: List[Path]) -> Dict[str, int]:
    """Ingest a list of local PDF file paths.

    Steps: copy into docs_dir, extract pages, chunk, persist, update manifest,
    rebuild TF-IDF index. Returns counts.
    """
    ensure_data_dirs()
    total_chunks = 0
    ingested: List[str] = []

    for src in file_paths:
        dst = docs_dir() / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        pages = extract_pdf_pages(dst)
        md5 = compute_md5(dst)
        doc_id = dst.stem
        upsert_document(doc_id=doc_id, filename=dst.name, md5=md5, pages=len(pages))
        chunks = build_chunks(doc_id=doc_id, pages=pages)
        persist_chunks(doc_id, chunks)
        total_chunks += len(chunks)
        ingested.append(doc_id)

    # Rebuild lexical index after batch
    build_index_from_all_chunks()
    return {"docs": len(ingested), "chunks": total_chunks}


