from pathlib import Path
import fitz

from backend.index.store import ensure_data_dirs, docs_dir
from backend.ingestion.extract import extract_pdf_pages
from backend.ingestion.manifest import compute_md5, upsert_document

ensure_data_dirs()

# Create a tiny PDF
pdf_path = docs_dir() / "smoke.pdf"
if not pdf_path.exists():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. Introduction\nThis is a smoke test PDF for extraction.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "2. Methods\nWe describe methods here.")
    doc.save(str(pdf_path))
    doc.close()

# Extract
pages = extract_pdf_pages(pdf_path)
print(f"Extracted {len(pages)} pages")
for p in pages:
    print({"page": p.page_index, "headings": p.heading_candidates[:3], "len": len(p.text)})

# Update manifest
md5 = compute_md5(pdf_path)
entry = upsert_document(doc_id="smoke-pdf", filename=pdf_path.name, md5=md5, pages=len(pages))
print("Manifest updated:", entry)
