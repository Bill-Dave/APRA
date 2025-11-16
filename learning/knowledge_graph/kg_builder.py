# learning/knowledge_graph/kg_builder.py
"""
KG builder: lightweight, dependency-tolerant tools to construct a knowledge graph
from text chunks. Persists graphs under ./data/kg/<book_id>.json by default.

Primary functions:
- build_kg_from_chunks(book_id, chunks, use_spacy=True)
- save_kg(book_id, kg, path=None)
- load_kg(book_id, path=None)

KGGraph is a simple wrapper around networkx if available, otherwise a dict-based graph.
"""
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import logging
from pathlib import Path
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

try:
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:
    _HAS_NX = False

try:
    import spacy  # type: ignore
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

KG_ROOT = Path(os.getenv("APRA_KG_ROOT", "./data/kg"))
KG_ROOT.mkdir(parents=True, exist_ok=True)


class KGGraph:
    """
    Minimal KG wrapper. If networkx is available, uses a DiGraph; otherwise uses dict storage.
    Nodes: id -> {"label": str, "type": str, "meta": {...}}
    Edges: (src, dst) -> {"rel": str, "weight": float, "meta": {...}}
    """

    def __init__(self):
        if _HAS_NX:
            self._g = nx.DiGraph()
        else:
            self._nodes = {}  # id -> attrs
            self._adj = defaultdict(list)  # src -> list of (dst, attrs)

    def add_node(self, node_id: str, label: str, ntype: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        meta = meta or {}
        attrs = {"label": label, "type": ntype or "entity", "meta": meta}
        if _HAS_NX:
            self._g.add_node(node_id, **attrs)
        else:
            self._nodes[node_id] = attrs

    def add_edge(self, src: str, dst: str, rel: str = "related_to", weight: float = 1.0, meta: Optional[Dict[str, Any]] = None):
        meta = meta or {}
        attrs = {"rel": rel, "weight": float(weight), "meta": meta}
        if _HAS_NX:
            if self._g.has_edge(src, dst):
                # increment weight and merge meta
                existing = self._g[src][dst]
                existing["weight"] = existing.get("weight", 0.0) + attrs["weight"]
                existing_meta = existing.get("meta", {})
                existing_meta.update(attrs["meta"])
                existing["meta"] = existing_meta
            else:
                self._g.add_edge(src, dst, **attrs)
        else:
            # append or merge
            found = False
            for i, (d, a) in enumerate(self._adj.get(src, [])):
                if d == dst and a.get("rel") == rel:
                    a["weight"] = a.get("weight", 0.0) + attrs["weight"]
                    a["meta"].update(attrs.get("meta", {}))
                    found = True
                    break
            if not found:
                self._adj[src].append((dst, attrs))

    def nodes(self) -> List[Tuple[str, Dict[str, Any]]]:
        if _HAS_NX:
            return list(self._g.nodes(data=True))
        else:
            return [(nid, dict(attrs)) for nid, attrs in self._nodes.items()]

    def edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        if _HAS_NX:
            return [(u, v, dict(d)) for u, v, d in self._g.edges(data=True)]
        else:
            out = []
            for src, lst in self._adj.items():
                for dst, attrs in lst:
                    out.append((src, dst, dict(attrs)))
            return out

    def to_dict(self) -> Dict[str, Any]:
        return {"nodes": [{ "id": n, **attrs} for n, attrs in self.nodes()],
                "edges": [{ "src": u, "dst": v, **attrs} for u, v, attrs in self.edges()]}

    def merge(self, other: "KGGraph"):
        # merge nodes and edges; simple behavior
        for nid, attrs in other.nodes():
            if _HAS_NX:
                if not self._g.has_node(nid):
                    self._g.add_node(nid, **attrs)
            else:
                if nid not in self._nodes:
                    self._nodes[nid] = attrs
        for u, v, attrs in other.edges():
            self.add_edge(u, v, rel=attrs.get("rel"), weight=attrs.get("weight", 1.0), meta=attrs.get("meta"))

    def neighbors(self, node_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        if _HAS_NX:
            if not self._g.has_node(node_id):
                return []
            return [(nbr, dict(self._g.nodes[nbr])) for nbr in self._g.successors(node_id)]
        else:
            out = []
            for dst, attrs in self._adj.get(node_id, []):
                node_attrs = self._nodes.get(dst, {})
                out.append((dst, dict(node_attrs)))
            return out

    def degree(self, node_id: str) -> int:
        if _HAS_NX:
            if not self._g.has_node(node_id):
                return 0
            return int(self._g.degree(node_id))
        else:
            deg = len(self._adj.get(node_id, []))
            # count inbound edges
            for src, lst in self._adj.items():
                for dst, _ in lst:
                    if dst == node_id:
                        deg += 1
            return deg


def _extract_entities_spacy(text: str, nlp) -> List[Tuple[str,str]]:
    ents = []
    try:
        doc = nlp(text[:10000])  # limit size
        for e in doc.ents:
            ents.append((e.text.strip(), e.label_))
    except Exception:
        logger.exception("spacy extraction failed")
    return ents


def _extract_entities_basic(text: str) -> List[Tuple[str,str]]:
    """
    Very rough heuristic: capitalized phrases as Named Entities.
    Returns list of (phrase, 'PROPN')
    """
    import re
    candidates = set()
    # look for sequences of capitalized words
    for match in re.finditer(r'\b([A-Z][a-z]{1,}\b(?:\s+[A-Z][a-z]{1,}\b)*)', text):
        phrase = match.group(1).strip()
        if len(phrase) > 1 and len(phrase.split()) <= 4:
            candidates.add(phrase)
    return [(c, "PROPN") for c in sorted(candidates)]


def _simple_relation_extraction(text: str, entities: List[Tuple[str,str]]) -> List[Tuple[str,str,str]]:
    """
    Naive co-occurrence relation: if two entities appear in same sentence, create relation.
    Relation label is 'cooccurs'.
    """
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', text) if s.strip()]
    relations = []
    ent_texts = [e[0] for e in entities]
    for s in sentences:
        found = [et for et in ent_texts if et in s]
        if len(found) >= 2:
            for i in range(len(found)):
                for j in range(i+1, len(found)):
                    relations.append((found[i], found[j], "cooccurs"))
    return relations


def _node_id_for_label(label: str) -> str:
    # deterministic id for an entity label
    h = hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]
    safe = label.replace(" ", "_")[:40]
    return f"n_{safe}_{h}"


def build_kg_from_chunks(book_id: str, chunks: List[Dict[str, Any]], use_spacy: bool = True) -> KGGraph:
    """
    Build a KGGraph for a book from provided chunks.
    chunks: list of {id, text}
    Returns KGGraph instance.
    """
    logger.info("Building KG for book %s (%d chunks)", book_id, len(chunks))
    kg = KGGraph()
    nlp = None
    if use_spacy and _HAS_SPACY:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            logger.exception("Failed to load spaCy model, falling back to basic extractor")
            nlp = None

    for c in chunks:
        text = c.get("text", "") or ""
        entities = _extract_entities_spacy(text, nlp) if nlp else _extract_entities_basic(text)
        # add nodes
        for ent_text, ent_type in entities:
            nid = _node_id_for_label(ent_text)
            kg.add_node(nid, label=ent_text, ntype=ent_type, meta={"source_chunk": c.get("id")})
        # relations
        rels = _simple_relation_extraction(text, entities)
        for a, b, rel in rels:
            na = _node_id_for_label(a)
            nb = _node_id_for_label(b)
            kg.add_edge(na, nb, rel=rel, weight=1.0, meta={"source_chunk": c.get("id")})
    # persist
    save_kg(book_id, kg)
    return kg


def _kg_path_for(book_id: str, path: Optional[str] = None) -> Path:
    p = Path(path) if path else KG_ROOT / f"{book_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_kg(book_id: str, kg: KGGraph, path: Optional[str] = None) -> str:
    p = _kg_path_for(book_id, path=path)
    data = kg.to_dict()
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved KG for book %s to %s", book_id, p)
    return str(p)


def load_kg(book_id: str, path: Optional[str] = None) -> Optional[KGGraph]:
    p = _kg_path_for(book_id, path=path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    kg = KGGraph()
    # restore nodes
    for n in data.get("nodes", []):
        nid = n.get("id")
        label = n.get("label") or n.get("meta", {}).get("label") or nid
        ntype = n.get("type") or n.get("meta", {}).get("type")
        meta = n.get("meta", {}) or {}
        kg.add_node(nid, label=label, ntype=ntype, meta=meta)
    for e in data.get("edges", []):
        kg.add_edge(e.get("src"), e.get("dst"), rel=e.get("rel"), weight=e.get("weight", 1.0), meta=e.get("meta", {}))
    return kg