# learning/knowledge_graph/__init__.py
"""
Knowledge Graph module - public exports.
"""
from .kg_builder import build_kg_from_chunks, save_kg, load_kg, KGGraph
from .kg_updater import add_entities_relations, merge_graphs, tag_provenance
from .kg_query import query_neighbors, find_path, query_by_label, top_entities, export_subgraph

__all__ = [
    "build_kg_from_chunks",
    "save_kg",
    "load_kg",
    "KGGraph",
    "add_entities_relations",
    "merge_graphs",
    "tag_provenance",
    "query_neighbors",
    "find_path",
    "query_by_label",
    "top_entities",
    "export_subgraph",
]