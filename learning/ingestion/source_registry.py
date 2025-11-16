# learning/ingestion/source_registry.py
"""
Registry and validation utilities for ingestion sources.

Responsibilities:
- Maintain allowed source types and domain allowlists/deny-lists
- Validate operator-supplied license/consent flags
- Normalize source descriptors used by ingestors
"""

from typing import Dict, Any, List
import hashlib
import logging
import urllib.parse

logger = logging.getLogger(__name__)

# Minimal defaults; operator may update at runtime
ALLOWED_SCHEMES = {"file", "http", "https"}
ALLOWED_PUBLIC_DOMAINS = {
    "gutenberg.org",
    "archive.org",
    "arxiv.org",
    "wikisource.org",
}

class SourceValidationError(Exception):
    pass

def normalize_source_descriptor(source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a source descriptor dict into canonical shape.
    Expected keys: { "type": "upload"|"url", "uri": str, "title": str|None, "license_confirmed": bool|None }
    """
    s = dict(source)
    t = s.get("type")
    uri = s.get("uri")
    if not t or not uri:
        raise SourceValidationError("source must include 'type' and 'uri'")
    if t not in ("upload", "url"):
        raise SourceValidationError(f"unsupported source type: {t}")
    s["uri"] = uri.strip()
    s["title"] = s.get("title") or None
    s["license_confirmed"] = bool(s.get("license_confirmed", False))
    return s

def is_public_domain_url(url: str) -> bool:
    try:
        p = urllib.parse.urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        host = p.hostname or ""
        host = host.lower()
        for dom in ALLOWED_PUBLIC_DOMAINS:
            if host.endswith(dom):
                return True
        return False
    except Exception:
        return False

def requires_license_confirmation(source: Dict[str, Any]) -> bool:
    """
    Return True if we must require operator license confirmation before ingesting the URL.
    - Public-domain hosts are allowed without extra confirmation.
    - Other hosts require explicit `license_confirmed=True`.
    """
    t = source.get("type")
    uri = source.get("uri", "")
    if t == "upload":
        # upload is considered operator-provided; still require a truthy flag if sensitive
        return False
    if t == "url":
        return not is_public_domain_url(uri)

def compute_checksum_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()