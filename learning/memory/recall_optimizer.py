# learning/memory/recall_optimizer.py
"""
Recall optimizer: schedules items for review and selects items for testing.

Features:
- Implements a simple SM-2 like spaced repetition scheduler (ease-factor based)
- Tracks per-item history in episodic memory (via append_event) or local sqlite
- select_items_for_review(count, cutoff_days) returns prioritized chunk ids
"""

from typing import List, Dict, Any, Optional
import sqlite3
import os
import datetime
import threading
import logging
import math
import json

logger = logging.getLogger(__name__)
DB_PATH = os.getenv("APRA_RECALL_DB", "./data/recall.db")
_LOCK = threading.Lock()


def _ensure_db():
    dbp = os.path.splitext(DB_PATH)[0]
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recall_items (
            id TEXT PRIMARY KEY,
            chunk_id TEXT,
            last_review TEXT,
            interval_days REAL,
            ease REAL,
            repetition INTEGER,
            score REAL,
            metadata_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def register_item(chunk_id: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Ensure an item exists in the recall DB.
    """
    _ensure_db()
    with _LOCK:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id FROM recall_items WHERE id = ?", (chunk_id,))
        if cur.fetchone():
            conn.close()
            return
        now = datetime.datetime.utcnow().isoformat() + "Z"
        cur.execute(
            "INSERT INTO recall_items (id, chunk_id, last_review, interval_days, ease, repetition, score, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chunk_id, chunk_id, now, 0.0, 2.5, 0, 0.0, json.dumps(metadata or {})),
        )
        conn.commit()
        conn.close()


def record_review(chunk_id: str, quality: float):
    """
    Record a review result for chunk_id.
    quality: 0.0 - 5.0 (as in SM-2)
    Updates interval, repetition and ease based on simplified SM-2 rules.
    """
    _ensure_db()
    with _LOCK:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT repetition, ease, interval_days FROM recall_items WHERE id = ?", (chunk_id,))
        row = cur.fetchone()
        if not row:
            # auto-register then call recursively
            register_item(chunk_id)
            conn.close()
            return record_review(chunk_id, quality)
        repetition, ease, interval = row
        if quality < 3.0:
            repetition = 0
            interval = 1.0
        else:
            repetition += 1
            if repetition == 1:
                interval = 1.0
            elif repetition == 2:
                interval = 6.0
            else:
                interval = max(1.0, interval * ease)
        # update ease factor
        ease = max(1.3, ease + 0.1 - (5.0 - quality) * (0.08 + (5.0 - quality) * 0.02))
        # compute next review timestamp
        next_review = (datetime.datetime.utcnow() + datetime.timedelta(days=interval)).isoformat() + "Z"
        # update score (inverse of interval for prioritization)
        score = 1.0 / max(0.01, interval)
        cur.execute(
            "UPDATE recall_items SET repetition = ?, ease = ?, interval_days = ?, last_review = ?, score = ? WHERE id = ?",
            (repetition, ease, interval, next_review, score, chunk_id),
        )
        conn.commit()
        conn.close()


def select_items_for_review(count: int = 10, due_within_days: Optional[float] = None) -> List[str]:
    """
    Select items prioritized by score and due time.
    If due_within_days is provided, filter items with next_review <= now + due_within_days.
    """
    _ensure_db()
    now = datetime.datetime.utcnow()
    cutoff = None
    if due_within_days is not None:
        cutoff = (now + datetime.timedelta(days=due_within_days)).isoformat() + "Z"
    with _LOCK:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        if cutoff:
            cur.execute("SELECT id, chunk_id, last_review, score FROM recall_items WHERE last_review <= ? ORDER BY score DESC LIMIT ?", (cutoff, count))
        else:
            cur.execute("SELECT id, chunk_id, last_review, score FROM recall_items ORDER BY score DESC LIMIT ?", (count,))
        rows = cur.fetchall()
        conn.close()
    return [r[1] for r in rows]


def remove_item(chunk_id: str):
    _ensure_db()
    with _LOCK:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM recall_items WHERE id = ?", (chunk_id,))
        conn.commit()
        conn.close()
```0