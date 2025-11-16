# learning/memory/consolidation_unit.py
"""
Consolidation unit: cluster micro-summaries into concept nodes and create meso/macro summaries.

Approach:
- Fetch micro summaries (via semantic_memory)
- Optionally use embedding vectors (if available) to cluster (KMeans or agglomerative)
- Produce merged summaries for each cluster (simple concatenation + optional LLM summarization hook)
- Store consolidated summaries back into semantic_memory as 'concept' level

Note: Uses numpy and sklearn if available; otherwise falls back to heuristic clustering by length.
"""
from typing import List, Dict, Any, Optional
import logging
import math
import statistics

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    from sklearn.cluster import KMeans  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

from learning.memory.semantic_memory import get_summary_by_chunk, upsert_summary, list_chunks_by_book, get_chunk, get_embedding_by_chunk


def _gather_summaries_for_book(book_id: str, level_filter: str = "micro") -> List[Dict[str, Any]]:
    chunks = list_chunks_by_book(book_id, limit=10000)
    summaries = []
    for c in chunks:
        ch_summ = get_summary_by_chunk(c["id"])
        for s in ch_summ:
            if s.get("level") == level_filter or (level_filter == "micro" and s.get("level") is None):
                summaries.append({"chunk": c, "summary": s})
    return summaries


def _compute_vectors(items: List[Dict[str, Any]]) -> Optional[List[List[float]]]:
    vecs = []
    for it in items:
        emb = get_embedding_by_chunk(it["chunk"]["id"])
        if emb and emb.get("vector"):
            vecs.append(emb["vector"])
        else:
            vecs.append(None)
    if all(v is None for v in vecs):
        return None
    # replace None with average vector if possible
    if _HAS_NUMPY:
        vecs_np = []
        avg = None
        for v in vecs:
            if v is not None:
                vecs_np.append(np.array(v))
        if vecs_np:
            avg = np.mean(vecs_np, axis=0)
        for i, v in enumerate(vecs):
            if v is None:
                vecs[i] = avg.tolist() if avg is not None else [0.0] * len(vecs_np[0])
        return vecs
    else:
        # fallback: map None to zeros of length of first non-None
        first = next((v for v in vecs if v is not None), None)
        if first is None:
            return None
        dim = len(first)
        return [v if v is not None else [0.0] * dim for v in vecs]


def consolidate_book(book_id: str, n_clusters: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Consolidate micro summaries for a book into concept summaries.

    Returns list of concept summary records.
    """
    items = _gather_summaries_for_book(book_id, level_filter="micro")
    if not items:
        return []

    vectors = _compute_vectors(items)
    num_items = len(items)
    if n_clusters is None:
        # heuristic: sqrt of items, bounded
        n_clusters = max(1, min(50, int(math.sqrt(num_items))))
    logger.info("Consolidating %d micro-summaries into %d clusters", num_items, n_clusters)

    clusters = None
    if vectors is not None and _HAS_SKLEARN:
        try:
            km = KMeans(n_clusters=n_clusters, random_state=42)
            km.fit(vectors)
            labels = km.labels_
            clusters = {}
            for idx, lab in enumerate(labels):
                clusters.setdefault(int(lab), []).append(items[idx])
        except Exception:
            logger.exception("KMeans failed")
            clusters = None

    if clusters is None:
        # fallback: simple round-robin grouping
        clusters = {}
        for i, it in enumerate(items):
            k = i % n_clusters
            clusters.setdefault(k, []).append(it)

    concept_summaries = []
    for k, group in clusters.items():
        texts = [g["summary"]["summary"] or g["chunk"]["text_ptr"] or "" for g in group]
        merged = "\n\n".join(t for t in texts if t)
        # simple heuristic summary: take first 2 sentences of merged or truncate
        preview = merged.strip().split("\n")
        excerpt = "\n".join(preview[:3])
        concept_id = f"concept-{book_id}-{k}"
        record = {
            "id": concept_id,
            "chunk_ids": [g["chunk"]["id"] for g in group],
            "book_id": book_id,
            "level": "concept",
            "summary": excerpt,
            "created_at": None,
        }
        # store as a summary back to semantic memory
        upsert_summary({
            "id": record["id"],
            "chunk_id": record["chunk_ids"][0],
            "level": "concept",
            "summary": record["summary"],
            "prompt": "consolidation_unit:merged",
            "model": "local-consolidator",
        })
        concept_summaries.append(record)
    logger.info("Created %d concept summaries for book %s", len(concept_summaries), book_id)
    return concept_summaries