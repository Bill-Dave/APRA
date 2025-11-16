# learning/ingestion/document_cleaner.py
"""
Extract plain text from raw document files.
Supports: .txt, .md, .html, .pdf (basic), .epub (basic)
Uses lightweight libraries if available; falls back to simple heuristics.

Note: Production deployments may prefer specialized extraction tools (tika, poppler, ebooklib).
This module is intentionally dependency-tolerant.
"""

import os
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# optional imports (guarded)
try:
    from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
except Exception:
    pdf_extract_text = None

try:
    import ebooklib
    from ebooklib import epub  # type: ignore
except Exception:
    epub = None

def extract_text_from_file(path: str) -> Tuple[str, dict]:
    """
    Return (clean_text, metadata)
    metadata may contain page_count, title, author, content_type
    """
    p = Path(path)
    suffix = p.suffix.lower()
    metadata = {"content_type": suffix}
    if suffix in (".txt", ".md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        return text, metadata
    if suffix in (".html", ".htm"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        # strip tags minimally
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(text, "html.parser")
            clean = soup.get_text(separator="\n")
            return clean, metadata
        except Exception:
            # fallback: crude removal
            import re
            clean = re.sub(r"<[^>]+>", "", text)
            return clean, metadata
    if suffix == ".pdf":
        if pdf_extract_text:
            try:
                txt = pdf_extract_text(str(p))
                metadata["page_count"] = None
                return txt, metadata
            except Exception:
                logger.exception("pdfminer extraction failed, falling back")
        # fallback: binary attempt (not great)
        return p.read_bytes().decode("latin-1", errors="ignore"), metadata
    if suffix == ".epub":
        if epub:
            try:
                book = epub.read_epub(str(p))
                items = []
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        items.append(item.get_content().decode("utf-8", errors="ignore"))
                text = "\n\n".join(items)
                return text, metadata
            except Exception:
                logger.exception("epub extraction failed, falling back")
        return p.read_text(encoding="utf-8", errors="ignore"), metadata
    # default fallback
    try:
        return p.read_text(encoding="utf-8", errors="ignore"), metadata
    except Exception:
        return p.read_bytes().decode("latin-1", errors="ignore"), metadata

def clean_text_basic(text: str) -> str:
    """
    Basic cleaning: normalize whitespace, remove nulls, trim repeated line breaks.
    """
    import re
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    # remove NULLs and control characters
    text = "".join(ch for ch in text if ord(ch) >= 9)
    # collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # strip leading/trailing
    text = text.strip()
    return text