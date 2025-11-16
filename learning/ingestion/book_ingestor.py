# learning/ingestion/book_ingestor.py
"""
Ingest uploaded files (PDF, EPUB, TXT). Responsible for:
- Storing raw file to configured raw store path
- Emitting a Book record (minimal dict) with metadata
- Returning a job descriptor for downstream pipeline

This implementation uses local filesystem storage by default under ./data/raw/
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import datetime
import logging

from .source_registry import normalize_source_descriptor, compute_checksum_bytes

logger = logging.getLogger(__name__)

RAW_ROOT_DEFAULT = os.getenv("APRA_RAW_ROOT", "./data/raw")

def ensure_raw_root(path: str = RAW_ROOT_DEFAULT) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_uploaded_file(file_path: str, dest_name: Optional[str] = None, raw_root: str = RAW_ROOT_DEFAULT) -> Dict[str, Any]:
    """
    Accept a local path (path provided by upload handler), copy to raw store, and register.
    Returns a book record dict.
    """
    raw_root_p = ensure_raw_root(raw_root)
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"upload file not found: {file_path}")
    dest_name = dest_name or f"{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{src.name}"
    dest = raw_root_p / dest_name
    shutil.copy2(str(src), str(dest))
    with dest.open("rb") as f:
        data = f.read()
    checksum = compute_checksum_bytes(data)
    book = {
        "id": f"book-{checksum[:12]}",
        "title": src.stem,
        "source_type": "upload",
        "raw_uri": str(dest.resolve()),
        "checksum": checksum,
        "ingested_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "raw_stored",
        "size_bytes": len(data),
    }
    logger.info("Saved uploaded file to raw store: %s", dest)
    return book

def register_book_from_descriptor(source: Dict[str, Any], uploaded_path: Optional[str] = None, raw_root: str = RAW_ROOT_DEFAULT) -> Dict[str, Any]:
    """
    High-level helper used by API route:
    - source: normalized source descriptor (type, uri, title, license_confirmed)
    - uploaded_path: optional path where the upload was temporarily saved (web frameworks)
    Returns a Book dict for downstream processing.
    """
    s = normalize_source_descriptor(source)
    if s["type"] != "upload":
        raise ValueError("register_book_from_descriptor only handles uploads")
    if not uploaded_path:
        raise ValueError("uploaded_path is required for upload sources")
    book = save_uploaded_file(uploaded_path, dest_name=None, raw_root=raw_root)
    # override title if provided
    if s.get("title"):
        book["title"] = s["title"]
    # attach license flag
    book["license_confirmed"] = s.get("license_confirmed", False)
    return book