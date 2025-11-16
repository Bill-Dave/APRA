# learning/memory/episodic_memory.py
"""
Episodic memory: append-only event traces for learning pipeline runs.

Responsibilities:
- Append event traces (worker operations, ingestion events, tests, corrections)
- Query recent events, filter by type or time window
- Export / replay traces for audit and replayability

Storage format: JSONL file (each line is a JSON record)
"""
from pathlib import Path
import json
import os
import threading
import datetime
from typing import Dict, Any, Iterable, List, Optional
import logging

logger = logging.getLogger(__name__)

EPISODIC_LOG = Path(os.getenv("APRA_EPISODIC_LOG", "./data/episodic.log"))
_LOCK = threading.Lock()


def _ensure_log():
    EPISODIC_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not EPISODIC_LOG.exists():
        EPISODIC_LOG.write_text("", encoding="utf-8")


def append_event(event_type: str, payload: Dict[str, Any], actor: Optional[str] = None) -> Dict[str, Any]:
    """
    Append an event to the episodic log and return the record.
    """
    _ensure_log()
    rec = {
        "id": f"evt-{int(datetime.datetime.utcnow().timestamp()*1000)}",
        "type": event_type,
        "actor": actor or "system",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "payload": payload,
    }
    line = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
    with _LOCK:
        with EPISODIC_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return rec


def iter_events(reverse: bool = True) -> Iterable[Dict[str, Any]]:
    """
    Iterate over events. If reverse=True, yields newest-first.
    """
    _ensure_log()
    with EPISODIC_LOG.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    if reverse:
        lines = reversed(lines)
    for line in lines:
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except Exception:
            logger.exception("Malformed episodic log line: %s", line)
            continue


def query_events(event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Simple query: filter by event_type (if provided) and return up to limit records (newest-first).
    """
    results = []
    for rec in iter_events(reverse=True):
        if event_type and rec.get("type") != event_type:
            continue
        results.append(rec)
        if len(results) >= limit:
            break
    return results


def export_events(out_path: str, event_type: Optional[str] = None, limit: Optional[int] = None) -> str:
    """
    Export filtered events to out_path (JSON array).
    """
    items = []
    for rec in iter_events(reverse=False):
        if event_type and rec.get("type") != event_type:
            continue
        items.append(rec)
        if limit and len(items) >= limit:
            break
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, indent=2), encoding="utf-8")
    logger.info("Exported %d events to %s", len(items), p)
    return str(p)