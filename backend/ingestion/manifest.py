from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

from backend.index.store import manifests_dir, write_json, read_json, ensure_data_dirs


_MANIFEST_FILE = manifests_dir() / "manifest.json"


@dataclass
class DocumentEntry:
    doc_id: str
    filename: str
    md5: str
    pages: int


def _load_manifest() -> Dict[str, DocumentEntry]:
    ensure_data_dirs()
    raw = read_json(_MANIFEST_FILE, default={})
    out: Dict[str, DocumentEntry] = {}
    for k, v in raw.items():
        out[k] = DocumentEntry(**v)
    return out


def _save_manifest(data: Dict[str, DocumentEntry]) -> None:
    serializable = {k: asdict(v) for k, v in data.items()}
    write_json(_MANIFEST_FILE, serializable)


def compute_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    m = hashlib.md5()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            m.update(chunk)
    return m.hexdigest()


def upsert_document(doc_id: str, filename: str, md5: str, pages: int) -> DocumentEntry:
    manifest = _load_manifest()
    entry = DocumentEntry(doc_id=doc_id, filename=filename, md5=md5, pages=pages)
    manifest[doc_id] = entry
    _save_manifest(manifest)
    return entry


def get_document(doc_id: str) -> Optional[DocumentEntry]:
    manifest = _load_manifest()
    return manifest.get(doc_id)


def all_documents() -> Dict[str, DocumentEntry]:
    return _load_manifest()


