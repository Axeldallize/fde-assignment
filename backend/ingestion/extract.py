from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import re

import fitz  # PyMuPDF


_RE_NUM_HEADING = re.compile(r"^(\d+[\.\)])+\s+.+")


@dataclass
class PageContent:
    page_index: int
    text: str
    heading_candidates: List[str]


def _detect_heading_candidates(page_text: str, max_candidates: int = 8) -> List[str]:
    candidates: List[str] = []
    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Numbered headings like "1.", "1.1", "2)"
        if _RE_NUM_HEADING.match(line):
            candidates.append(line)
            continue
        # ALL CAPS short lines (<= 80 chars) as likely headings
        if len(line) <= 80 and line.upper() == line and any(c.isalpha() for c in line):
            candidates.append(line)
            continue
        # Title-cased single-line words (heuristic)
        if len(line.split()) <= 8 and line[:1].isupper() and line == line.title():
            candidates.append(line)
        if len(candidates) >= max_candidates:
            break
    return candidates


def extract_pdf_pages(file_path: Path) -> List[PageContent]:
    doc = fitz.open(str(file_path))
    pages: List[PageContent] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text")
            headings = _detect_heading_candidates(text)
            pages.append(PageContent(page_index=i, text=text, heading_candidates=headings))
    finally:
        doc.close()
    return pages


