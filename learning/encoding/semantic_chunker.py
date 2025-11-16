# learning/encoding/semantic_chunker.py
"""
Semantic chunker that slices text into overlapping chunks by token count.

Exports:
- semantic_chunk(text, target_tokens=200, overlap=50, tokenizer=None)
  -> List[dict] with keys: id, level, start_token, end_token, text
"""
from typing import List, Optional, Dict, Any
import hashlib
import logging
import math

from .tokenizer import get_tokenizer, count_tokens

logger = logging.getLogger(__name__)


def _chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def semantic_chunk(
    text: str,
    target_tokens: int = 200,
    overlap: int = 50,
    tokenizer: Optional[Any] = None,
    level: str = "micro",
) -> List[Dict[str, Any]]:
    """
    Break `text` into overlapping chunks of approximately `target_tokens` tokens.

    Returns list of dicts:
      { id, level, start_token, end_token, text }
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    try:
        tokens = tokenizer.encode(text)
    except Exception:
        # tokenizer fallback: treat tokens as words
        tokens = text.split()

    if not tokens:
        return []

    stride = max(1, target_tokens - overlap)
    chunks = []
    num_tokens = len(tokens)
    idx = 0
    chunk_index = 0
    while idx < num_tokens:
        end = min(num_tokens, idx + target_tokens)
        segment_tokens = tokens[idx:end]
        # attempt decode, but be tolerant
        try:
            chunk_text = tokenizer.decode(segment_tokens)
        except Exception:
            # if tokens are words, join them
            if isinstance(segment_tokens, list):
                chunk_text = " ".join(str(t) for t in segment_tokens)
            else:
                chunk_text = str(segment_tokens)
        cid = f"{_chunk_id(chunk_text)}_{chunk_index}"
        chunks.append(
            {
                "id": cid,
                "level": level,
                "start_token": idx,
                "end_token": end,
                "text": chunk_text,
            }
        )
        chunk_index += 1
        idx += stride
    return chunks


def merge_chunks(chunks: List[Dict[str, Any]], max_tokens: int = 800, tokenizer: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Merge adjacent chunks conservatively so each merged chunk <= max_tokens.
    Useful to build meso-level chunks from micro chunks.
    """
    if not chunks:
        return []
    if tokenizer is None:
        tokenizer = get_tokenizer()

    merged = []
    buffer_text = []
    buffer_count = 0
    buffer_start = None
    buffer_level = chunks[0].get("level", "merged")
    idx = 0

    for c in chunks:
        t = c["text"]
        ccount = count_tokens(t)
        if buffer_count + ccount <= max_tokens:
            if buffer_start is None:
                buffer_start = c["start_token"]
            buffer_text.append(t)
            buffer_count += ccount
        else:
            merged_text = "\n\n".join(buffer_text)
            cid = _chunk_id(merged_text) + f"_m{idx}"
            merged.append({"id": cid, "level": buffer_level, "start_token": buffer_start, "end_token": c["end_token"], "text": merged_text})
            idx += 1
            # reset
            buffer_text = [t]
            buffer_count = ccount
            buffer_start = c["start_token"]
    # flush
    if buffer_text:
        merged_text = "\n\n".join(buffer_text)
        cid = _chunk_id(merged_text) + f"_m{idx}"
        merged.append({"id": cid, "level": buffer_level, "start_token": buffer_start or 0, "end_token": chunks[-1]["end_token"], "text": merged_text})
    return merged