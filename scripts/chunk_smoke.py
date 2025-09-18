from pathlib import Path
from backend.index.store import ensure_data_dirs, chunks_dir, docs_dir
from backend.ingestion.extract import extract_pdf_pages
from backend.ingestion.chunk import build_chunks, persist_chunks
from backend.ingestion.manifest import compute_md5, upsert_document

ensure_data_dirs()

pdf_path = docs_dir() / "smoke.pdf"
if not pdf_path.exists():
    raise SystemExit("smoke.pdf not found. Run scripts/pdf_extract_smoke.py first.")

pages = extract_pdf_pages(pdf_path)
chunks = build_chunks(doc_id="smoke-pdf", pages=pages, target_tokens=60, overlap_ratio=0.15)
print(f"Built {len(chunks)} chunks. Example: {chunks[0].chunk_id if chunks else N/A}")
paths = persist_chunks("smoke-pdf", chunks)
print("Wrote:", paths)
print("Chunk files present:", (chunks_dir() / "smoke-pdf.jsonl").exists())
