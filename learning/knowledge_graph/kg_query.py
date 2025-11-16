# learning/knowledge_graph/kg_query.py
"""
Simple KG query utilities.

Functions:
- query_neighbors(book_id, node_label_or_id, depth=1)
- find_path(book_id, source_label_or_id, target_label_or_id, max_hops=6)
- query_by_label(book_id, label_substring)
- top_entities(book_id, top_k=20)
- export_subgraph(book_id, node_ids, out_path)
"""
from typing import List, Tuple, Dict, Any, Optional
import logging
from .kg_builder import load_kg, _node_id_for_label
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_node_id(kg, label_or_id: str) -> Optional[str]:
    # if label_or_id looks like a node id, return directly if exists
    if label_or_id.startswith("n_"):
        # check existence
        nodes = {nid for nid, _ in kg.nodes()}
        if label_or_id in nodes:
            return label_or_id
    # else try match by label substring (first match)
    for nid, attrs in kg.nodes():
        lbl = attrs.get("label", "") or ""
        if label_or_id.lower() in lbl.lower():
            return nid
    # fallback deterministic id
    candidate = _node_id_for_label(label_or_id)
    for nid, _ in kg.nodes():
        if nid == candidate:
            return nid
    return None


def query_neighbors(book_id: str, node_label_or_id: str, depth: int = 1) -> Dict[str, Any]:
    kg = load_kg(book_id)
    if not kg:
        return {"error": "kg_not_found"}
    root = _resolve_node_id(kg, node_label_or_id)
    if not root:
        return {"error": "node_not_found"}
    visited = set([root])
    frontier = [root]
    result_nodes = []
    for d in range(depth):
        next_frontier = []
        for n in frontier:
            nbrs = kg.neighbors(n)
            for nid, attrs in nbrs:
                if nid not in visited:
                    visited.add(nid)
                    next_frontier.append(nid)
                    result_nodes.append({"id": nid, "attrs": attrs})
        frontier = next_frontier
        if not frontier:
            break
    return {"root": root, "depth": depth, "nodes": result_nodes}


def find_path(book_id: str, source_label_or_id: str, target_label_or_id: str, max_hops: int = 6) -> Dict[str, Any]:
    kg = load_kg(book_id)
    if not kg:
        return {"error": "kg_not_found"}
    src = _resolve_node_id(kg, source_label_or_id)
    dst = _resolve_node_id(kg, target_label_or_id)
    if not src or not dst:
        return {"error": "node_not_found", "src": src, "dst": dst}
    # if networkx available, use shortest_path; otherwise BFS
    try:
        if hasattr(kg, "_g"):
            import networkx as nx  # type: ignore
            G = kg._g
            if nx.has_path(G, src, dst):
                path = nx.shortest_path(G, src, dst)
                return {"path": path}
            else:
                return {"path": None}
    except Exception:
        logger.exception("networkx path lookup failed, falling back to BFS")
    # BFS fallback
    from collections import deque
    q = deque([[src]])
    seen = set([src])
    while q:
        path = q.popleft()
        node = path[-1]
        if node == dst:
            return {"path": path}
        if len(path) > max_hops:
            continue
        for nbr, _ in kg.neighbors(node):
            if nbr not in seen:
                seen.add(nbr)
                new_path = list(path)
                new_path.append(nbr)
                q.append(new_path)
    return {"path": None}


def query_by_label(book_id: str, label_substring: str, limit: int = 20) -> List[Dict[str, Any]]:
    kg = load_kg(book_id)
    if not kg:
        return []
    out = []
    for nid, attrs in kg.nodes():
        lbl = attrs.get("label", "") or ""
        if label_substring.lower() in lbl.lower():
            out.append({"id": nid, "label": lbl, "type": attrs.get("type"), "meta": attrs.get("meta")})
            if len(out) >= limit:
                break
    return out


def top_entities(book_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
    kg = load_kg(book_id)
    if not kg:
        return []
    degrees = []
    for nid, attrs in kg.nodes():
        deg = kg.degree(nid)
        degrees.append((deg, nid, attrs))
    degrees.sort(reverse=True)
    return [{"id": nid, "label": attrs.get("label"), "degree": deg} for deg, nid, attrs in degrees[:top_k]]


def export_subgraph(book_id: str, node_ids: List[str], out_path: Optional[str] = None) -> Dict[str, Any]:
    kg = load_kg(book_id)
    if not kg:
        return {"error": "kg_not_found"}
    # collect nodes and edges reachable within 1 hop of provided node_ids
    nodes = {}
    edges = []
    for nid in node_ids:
        # try resolve by label
        resolved = _resolve_node_id(kg, nid) or nid
        for n, attrs in kg.nodes():
            if n == resolved:
                nodes[n] = attrs
        for dst, attrs in kg.neighbors(resolved):
            edges.append({"src": resolved, "dst": dst, **attrs})
            if dst not in nodes:
                # include dst attrs
                for n2, a2 in kg.nodes():
                    if n2 == dst:
                        nodes[dst] = a2
    out = {"nodes": [{"id": n, **attrs} for n, attrs in nodes.items()], "edges": edges}
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return {"path": str(p)}
    return out
```0