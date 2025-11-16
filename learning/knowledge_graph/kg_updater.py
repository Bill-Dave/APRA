# learning/knowledge_graph/kg_updater.py
"""
KG updater utilities: incremental updating, merging, tagging provenance.

Functions:
- add_entities_relations(book_id, entities, relations, provenance)
- merge_graphs(target_book_id, source_book_id, out_book_id=None)
- tag_provenance(book_id, node_or_edge_id, provenance_tag)
"""
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
import json

from .kg_builder import load_kg, save_kg, KGGraph, _node_id_for_label, KG_ROOT

logger = logging.getLogger(__name__)


def add_entities_relations(book_id: str, entities: List[Tuple[str, str]], relations: List[Tuple[str, str, str]], provenance: Optional[Dict[str, Any]] = None) -> str:
    """
    Add or update nodes and edges to an existing KG for book_id.
    entities: list of (label, type)
    relations: list of (label_a, label_b, rel)
    provenance: optional dict stored on created edges/nodes meta
    Returns path to saved KG.
    """
    kg = load_kg(book_id) or KGGraph()
    for label, ntype in entities:
        nid = _node_id_for_label(label)
        kg.add_node(nid, label=label, ntype=ntype, meta={"provenance": provenance or {}})
    for a, b, rel in relations:
        na = _node_id_for_label(a)
        nb = _node_id_for_label(b)
        kg.add_edge(na, nb, rel=rel, weight=1.0, meta={"provenance": provenance or {}})
    path = save_kg(book_id, kg)
    logger.info("Added %d entities and %d relations to KG %s", len(entities), len(relations), book_id)
    return path


def merge_graphs(target_book_id: str, source_book_id: str, out_book_id: Optional[str] = None) -> str:
    """
    Merge KG from source_book_id into target_book_id. If out_book_id is provided, write merged result there;
    otherwise overwrite target_book_id.
    Returns path to saved merged KG.
    """
    kg_t = load_kg(target_book_id) or KGGraph()
    kg_s = load_kg(source_book_id)
    if not kg_s:
        raise FileNotFoundError(f"Source KG not found: {source_book_id}")
    kg_t.merge(kg_s)
    out_id = out_book_id or target_book_id
    path = save_kg(out_id, kg_t)
    logger.info("Merged KG %s into %s -> %s", source_book_id, target_book_id, out_id)
    return path


def tag_provenance(book_id: str, node_or_edge_identifier: str, provenance_tag: Dict[str, Any]) -> str:
    """
    Attach provenance_tag to a node or edge. node_or_edge_identifier format:
      - node:<node_id>
      - edge:<src>::<dst>::<rel>  (rel optional)
    """
    kg = load_kg(book_id)
    if not kg:
        raise FileNotFoundError(f"KG not found: {book_id}")

    # naive implementation: modify node meta or edge meta then save
    if node_or_edge_identifier.startswith("node:"):
        nid = node_or_edge_identifier.split("node:", 1)[1]
        # attempt to add meta
        if hasattr(kg, "_g") and hasattr(kg._g, "nodes"):
            if kg._g.has_node(nid):
                data = kg._g.nodes[nid].get("meta", {})
                data.update({"provenance": provenance_tag})
                kg._g.nodes[nid]["meta"] = data
        else:
            if nid in kg._nodes:
                kg._nodes[nid]["meta"].update({"provenance": provenance_tag})
    elif node_or_edge_identifier.startswith("edge:"):
        _, rest = node_or_edge_identifier.split("edge:", 1)
        parts = rest.split("::")
        if len(parts) >= 2:
            src, dst = parts[0], parts[1]
            rel = parts[2] if len(parts) > 2 else None
            # find matching edge and update meta
            if hasattr(kg, "_g"):
                if kg._g.has_edge(src, dst):
                    e = kg._g[src][dst]
                    meta = e.get("meta", {})
                    meta.update({"provenance": provenance_tag})
                    e["meta"] = meta
            else:
                lst = kg._adj.get(src, [])
                for i, (d, attrs) in enumerate(lst):
                    if d == dst and (rel is None or attrs.get("rel") == rel):
                        attrs_meta = attrs.get("meta", {})
                        attrs_meta.update({"provenance": provenance_tag})
                        lst[i] = (d, attrs)
    else:
        raise ValueError("Invalid identifier format")
    path = save_kg(book_id, kg)
    logger.info("Tagged provenance on %s in KG %s", node_or_edge_identifier, book_id)
    return path