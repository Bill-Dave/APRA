# learning/memory/long_term_store.py
"""
Long-term persistent raw storage and metadata registry.

Responsibilities:
- Persist raw uploaded/downloaded files in immutable storage (by checksum)
- Maintain a JSON metadata index for books and raw artifacts
- Provide snapshot and export utilities (read-only primary API)
- Minimal dependency footprint (uses stdlib only)

Storage layout (default):
./data/raw/                -> raw binary files (immutable)
./data/lt_index.json       -> metadata index (book_id -> metadata)

Note: This module intentionally avoids deleting raw artifacts; operations are append-only.
"""
from pathlib import Path
import json
import os
import threading
import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

RAW_ROOT = Path(os.getenv("APRA_RAW_ROOT", "./data/raw"))
INDEX_PATH = Path(os.getenv("APRA_LT_INDEX", "./data/lt_index.json"))
_LOCK = threading.Lock()


def _ensure_paths():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.write_text(json.dumps({"books": {}, "artifacts": {}}))


def _load_index() -> Dict[str, Any]:
    _ensure_paths()
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load long-term index; resetting")
        data = {"books": {}, "artifacts": {}}
        INDEX_PATH.write_text(json.dumps(data))
        return data


def _write_index(idx: Dict[str, Any]):
    tmp = INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    tmp.replace(INDEX_PATH)


def register_raw_artifact(raw_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Register an existing raw file path into the long-term index.
    raw_path should be an absolute or relative path to a file under RAW_ROOT.
    Returns the artifact metadata dict.
    """
    _ensure_paths()
    raw_path = str(raw_path)
    stat = Path(raw_path).stat()
    checksum = metadata.get("checksum") if metadata and "checksum" in metadata else None
    book_id = metadata.get("id") if metadata and "id" in metadata else f"book-{checksum[:12]}" if checksum else f"artifact-{int(stat.st_mtime)}"
    entry = {
        "id": book_id,
        "raw_uri": raw_path,
        "size_bytes": stat.st_size,
        "ingested_at": metadata.get("ingested_at") if metadata and "ingested_at" in metadata else datetime.datetime.utcnow().isoformat() + "Z",
        "metadata": metadata or {},
    }
    with _LOCK:
        idx = _load_index()
        idx["artifacts"][book_id] = entry
        _write_index(idx)
    logger.info("Registered raw artifact %s", book_id)
    return entry


def list_raw_artifacts() -> List[Dict[str, Any]]:
    idx = _load_index()
    return list(idx.get("artifacts", {}).values())


def register_book_record(book_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register a book record produced by ingestors.
    book_record is expected to include at least: id, title, raw_uri, checksum
    """
    if "id" not in book_record:
        raise ValueError("book_record must have an 'id'")
    with _LOCK:
        idx = _load_index()
        idx["books"][book_record["id"]] = book_record
        _write_index(idx)
    logger.info("Registered book %s", book_record["id"])
    return book_record


def get_book(book_id: str) -> Optional[Dict[str, Any]]:
    idx = _load_index()
    return idx.get("books", {}).get(book_id)


def snapshot_index(destination: str) -> str:
    """
    Export the current index to destination path (JSON). Returns the dest path.
    """
    idx = _load_index()
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    logger.info("Snapshot written to %s", dest)
    return str(dest)


def export_raw_to_directory(book_id: str, out_dir: str) -> Optional[str]:
    """
    Copy the raw artifact for book_id to out_dir and return path to copied file.
    """
    rec = get_book(book_id)
    if not rec:
        return None
    src = Path(rec["raw_uri"])
    if not src.exists():
        logger.error("Raw file missing for book %s: %s", book_id, src)
        return None
    dst_dir = Path(out_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    import shutil
    shutil.copy2(str(src), str(dst))
    logger.info("Exported %s to %s", src, dst)
    return str(dst)