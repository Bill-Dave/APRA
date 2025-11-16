# learning/memory/semantic_memory.py
"""
Semantic memory: stores chunk records, summaries, embeddings metadata and retrieval helpers.

Responsibilities:
- Persist chunk metadata (id, book_id, level, start/end, text pointer)
- Persist summary records (micro/meso/macro) with provenance
- Persist embedding metadata (embedding_id, chunk_id, model, vector_loc pointer optional)
- Provide simple retrieval by id, by book, or by metadata filters

Implementation:
- Uses a lightweight SQLite DB (stdlib sqlite3) for structured queries.
- Embedding vectors themselves are stored in separate vector index/backends (not here),
  but this module stores references and small cached vectors for quick heuristics.
"""
from typing import Optional, List, Dict, Any
import sqlite3
import os
import json
import threading
import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = Path(os.getenv("APRA_SEMANTIC_DB", "./data/semantic_memory.db"))
_LOCK = threading.Lock()


def _ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    # tables: chunks, summaries, embeddings
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            book_id TEXT,
            level TEXT,
            start_token INTEGER,
            end_token INTEGER,
            text_ptr TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            id TEXT PRIMARY KEY,
            chunk_id TEXT,
            level TEXT,
            summary TEXT,
            prompt TEXT,
            model TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            chunk_id TEXT,
            model TEXT,
            vector_json TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def upsert_chunk(chunk: Dict[str, Any]):
    """
    chunk: {id, book_id, level, start_token, end_token, text_ptr}
    """
    _ensure_db()
    with _LOCK:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO chunks (id, book_id, level, start_token, end_token, text_ptr, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk["id"],
                chunk.get("book_id"),
                chunk.get("level"),
                chunk.get("start_token"),
                chunk.get("end_token"),
                chunk.get("text_ptr"),
                datetime.datetime.utcnow().isoformat() + "Z",
            ),
        )
        conn.commit()
        conn.close()


def get_chunk(chunk_id: str) -> Optional[Dict[str, Any]]:
    _ensure_db()
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT id, book_id, level, start_token, end_token, text_ptr, created_at FROM chunks WHERE id = ?", (chunk_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "book_id": row[1],
        "level": row[2],
        "start_token": row[3],
        "end_token": row[4],
        "text_ptr": row[5],
        "created_at": row[6],
    }


def list_chunks_by_book(book_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    _ensure_db()
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT id, level, start_token, end_token, text_ptr FROM chunks WHERE book_id = ? ORDER BY rowid ASC LIMIT ?", (book_id, limit))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({"id": r[0], "level": r[1], "start_token": r[2], "end_token": r[3], "text_ptr": r[4]})
    return out


def upsert_summary(summary: Dict[str, Any]):
    """
    summary: {id, chunk_id, level, summary, prompt, model}
    """
    _ensure_db()
    with _LOCK:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO summaries (id, chunk_id, level, summary, prompt, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (summary["id"], summary["chunk_id"], summary.get("level"), summary.get("summary"), summary.get("prompt"), summary.get("model"), datetime.datetime.utcnow().isoformat() + "Z"),
        )
        conn.commit()
        conn.close()


def get_summary_by_chunk(chunk_id: str) -> List[Dict[str, Any]]:
    _ensure_db()
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT id, level, summary, prompt, model, created_at FROM summaries WHERE chunk_id = ?", (chunk_id,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({"id": r[0], "level": r[1], "summary": r[2], "prompt": r[3], "model": r[4], "created_at": r[5]})
    return out


def upsert_embedding(emb: Dict[str, Any]):
    """
    emb: {id, chunk_id, model, vector}  - vector can be a small cached list (json).
    """
    _ensure_db()
    vec_json = json.dumps(emb.get("vector")) if emb.get("vector") is not None else None
    with _LOCK:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO embeddings (id, chunk_id, model, vector_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (emb["id"], emb.get("chunk_id"), emb.get("model"), vec_json, datetime.datetime.utcnow().isoformat() + "Z"),
        )
        conn.commit()
        conn.close()


def get_embedding_by_chunk(chunk_id: str) -> Optional[Dict[str, Any]]:
    _ensure_db()
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT id, model, vector_json, created_at FROM embeddings WHERE chunk_id = ? ORDER BY created_at DESC LIMIT 1", (chunk_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    vec = json.loads(row[2]) if row[2] else None
    return {"id": row[0], "model": row[1], "vector": vec, "created_at": row[3]}