# learning/ingestion/web_ingestor.py
"""
Ingest a document from a URL. Behavior:
- Validate source descriptor and robots.txt
- If the URL requires license confirmation and none provided -> raise error
- Download the resource (streaming) and store in raw store
- Return a Book record for downstream processing
"""

import os
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from urllib.parse import urlparse
from .source_registry import normalize_source_descriptor, requires_license_confirmation, compute_checksum_bytes

logger = logging.getLogger(__name__)

RAW_ROOT_DEFAULT = os.getenv("APRA_RAW_ROOT", "./data/raw")
REQUEST_TIMEOUT = int(os.getenv("APRA_HTTP_TIMEOUT", "15"))
STREAM_CHUNK = 1024 * 64

def check_robots_allow(url: str, user_agent: str = "*") -> bool:
    """
    Minimal robots.txt check: fetch robots.txt and look for Disallow rules.
    This is a lightweight best-effort check; absence of robots.txt is treated as allowed.
    """
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.hostname}/robots.txt"
        r = requests.get(robots_url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return True
        text = r.text.splitlines()
        # super simple: if any "Disallow: /" present, block
        for line in text:
            line = line.strip().lower()
            if line.startswith("user-agent:") or line.startswith("disallow:"):
                # naive allow for now: only block if Disallow: /
                if line.startswith("disallow:") and line.split(":",1)[1].strip() == "/":
                    return False
        return True
    except Exception:
        # on errors, be conservative and allow fetch but log warning
        logger.warning("robots.txt check failed for %s", url, exc_info=True)
        return True

def download_to_raw(url: str, dest_name: Optional[str] = None, raw_root: str = RAW_ROOT_DEFAULT) -> Dict[str, Any]:
    root = Path(raw_root)
    root.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    filename = dest_name or (Path(parsed.path).name or "download")
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest = root / f"{timestamp}_{filename}"
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=STREAM_CHUNK):
                if chunk:
                    fh.write(chunk)
    with dest.open("rb") as f:
        data = f.read()
    checksum = compute_checksum_bytes(data)
    book = {
        "id": f"book-{checksum[:12]}",
        "title": Path(dest).stem,
        "source_type": "url",
        "raw_uri": str(dest.resolve()),
        "origin_url": url,
        "checksum": checksum,
        "ingested_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "raw_stored",
        "size_bytes": len(data),
    }
    logger.info("Downloaded and stored URL content: %s", dest)
    return book

def ingest_url(source: Dict[str, Any], allow_if_public: bool = True, raw_root: str = RAW_ROOT_DEFAULT) -> Dict[str, Any]:
    s = normalize_source_descriptor(source)
    if requires_license_confirmation(s) and not s.get("license_confirmed", False):
        raise PermissionError("This URL requires license confirmation before ingestion.")
    if not check_robots_allow(s["uri"]):
        raise PermissionError("robots.txt disallows fetching this resource.")
    return download_to_raw(s["uri"], dest_name=None, raw_root=raw_root)