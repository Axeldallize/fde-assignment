from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import tempfile


def _backend_root() -> Path:
    # backend/index/store.py → backend
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return _backend_root() / "data"


def docs_dir() -> Path:
    return data_dir() / "docs"


def chunks_dir() -> Path:
    return data_dir() / "chunks"


def index_dir() -> Path:
    return data_dir() / "index"


def manifests_dir() -> Path:
    return data_dir() / "manifests"


def ensure_data_dirs() -> None:
    for path in [data_dir(), docs_dir(), chunks_dir(), index_dir(), manifests_dir()]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write(path, text)


