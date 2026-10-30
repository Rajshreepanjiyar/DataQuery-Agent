"""
rag.py — Part A: Database Preparation (One-Time Process)

This module prepares and persists all database knowledge required by the query agent:
  1. Schema Embedding & FAISS Indexing — builds rich text representations of every table
     (name, columns, types, PKs, FKs, descriptions), embeds them with SentenceTransformers,
     stores the vectors in a FAISS IndexFlatIP index, and persists both the index and the
     metadata catalog to disk.
  2. Foreign-Key Relationship Graph — builds an adjacency-list graph of all tables connected
     by foreign keys with exact column mappings, and persists it to disk.

Run this script once (or whenever the schema changes) to regenerate all artifacts.
Artifacts produced:
  - schema.index           (FAISS vector index)
  - schema_documents.pkl   (table metadata catalog)
  - fk_graph.pkl           (foreign-key relationship graph)
"""

import os
import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer

from schema import get_complete_metadata, get_table_descriptions

# Paths for persisted artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "schema.index")
SCHEMA_DOCS_PATH = os.path.join(BASE_DIR, "schema_documents.pkl")
FK_GRAPH_PATH = os.path.join(BASE_DIR, "fk_graph.pkl")

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


def build_table_documents(schema, primary_keys, foreign_keys, descriptions):
    """
    Create a rich text document for every table using its name, columns, data types,
    primary keys, foreign keys, and domain description.
    Returns a list of dicts: {table_name, content, columns, primary_keys, foreign_keys_out, foreign_keys_in}.
    """
    # Pre-index foreign keys by child and parent table for fast lookup
    fk_by_child = {}
    fk_by_parent = {}
    for fk in foreign_keys:
        fk_by_child.setdefault(fk["child_table"], []).append(fk)
        fk_by_parent.setdefault(fk["parent_table"], []).append(fk)

    documents = []
    for table_name, columns in schema.items():
        desc = descriptions.get(table_name, "")
        pks = primary_keys.get(table_name, [])
        child_fks = fk_by_child.get(table_name, [])
        parent_fks = fk_by_parent.get(table_name, [])

        # Build rich text representation
        text_parts = [
            f"Table: {table_name}",
            f"Description: {desc}",
            "",
            "Columns:",
        ]
        col_info = []
        for col in columns:
            col_name = col[0]
            data_type = col[1]
            nullable = col[2]
            key = col[3]
            default = col[4]
            extra = col[5] if len(col) > 5 else ""
            line = f"  {col_name} {data_type}"
            if key == "PRI":
                line += " PRIMARY KEY"
            if key == "MUL":
                line += " FOREIGN KEY"
            if extra:
                line += f" {extra}"
            text_parts.append(line)
            col_info.append({
                "name": col_name,
                "type": data_type,
                "nullable": nullable,
                "key": key,
                "default": str(default) if default is not None else None,
                "extra": extra
            })

        if pks:
            text_parts.append(f"\nPrimary Key: {', '.join(pks)}")

        if child_fks:
            text_parts.append("\nForeign Keys (outgoing):")
            for fk in child_fks:
                text_parts.append(
                    f"  {fk['child_column']} -> {fk['parent_table']}.{fk['parent_column']}"
                )

        if parent_fks:
            text_parts.append("\nReferenced By (incoming):")
            for fk in parent_fks:
                text_parts.append(
                    f"  {fk['child_table']}.{fk['child_column']} -> {table_name}.{fk['parent_column']}"
                )

        content = "\n".join(text_parts)

        documents.append({
            "table_name": table_name,
            "content": content,
            "description": desc,
            "columns": col_info,
            "primary_keys": pks,
            "foreign_keys_out": [
                {"column": fk["child_column"], "ref_table": fk["parent_table"], "ref_column": fk["parent_column"]}
                for fk in child_fks
            ],
            "foreign_keys_in": [
                {"from_table": fk["child_table"], "from_column": fk["child_column"], "column": fk["parent_column"]}
                for fk in parent_fks
            ],
        })

    return documents


def build_faiss_index(documents, model):
    """
    Generate embeddings for all table documents and build a FAISS IndexFlatIP
    (cosine similarity via normalized vectors).
    Returns the FAISS index.
    """
    doc_texts = [doc["content"] for doc in documents]
    embeddings = model.encode(doc_texts, convert_to_numpy=True).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, dimension


def build_fk_graph(foreign_keys):
    """
    Build an adjacency-list graph of tables connected by foreign-key relationships.
    Each edge stores the exact column mappings for JOIN construction.

    Structure:
    {
        "table_a": {
            "table_b": [
                {"from_col": "col_x", "to_col": "col_y", "direction": "child_to_parent"}
            ]
        }
    }

    The graph is undirected — both directions are stored so BFS can traverse freely.
    """
    graph = {}

    for fk in foreign_keys:
        child = fk["child_table"]
        parent = fk["parent_table"]
        child_col = fk["child_column"]
        parent_col = fk["parent_column"]

        # child -> parent direction
        graph.setdefault(child, {})
        graph[child].setdefault(parent, [])
        edge = {
            "from_col": child_col,
            "to_col": parent_col,
            "direction": "child_to_parent",
            "child_table": child,
            "parent_table": parent,
            "child_column": child_col,
            "parent_column": parent_col
        }
        # Avoid duplicate edges
        if edge not in graph[child][parent]:
            graph[child][parent].append(edge)

        # parent -> child direction (reverse for undirected traversal)
        graph.setdefault(parent, {})
        graph[parent].setdefault(child, [])
        reverse_edge = {
            "from_col": parent_col,
            "to_col": child_col,
            "direction": "parent_to_child",
            "child_table": child,
            "parent_table": parent,
            "child_column": child_col,
            "parent_column": parent_col
        }
        if reverse_edge not in graph[parent][child]:
            graph[parent][child].append(reverse_edge)

    return graph


def persist_artifacts(index, documents, fk_graph):
    """Save FAISS index, schema documents, and FK graph to disk."""
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(SCHEMA_DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    with open(FK_GRAPH_PATH, "wb") as f:
        pickle.dump(fk_graph, f)


def load_artifacts():
    """Load persisted FAISS index, schema documents, and FK graph from disk."""
    if not all(os.path.exists(p) for p in [FAISS_INDEX_PATH, SCHEMA_DOCS_PATH, FK_GRAPH_PATH]):
        raise FileNotFoundError(
            "Persisted artifacts not found. Run `python rag.py` first to build them."
        )

    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(SCHEMA_DOCS_PATH, "rb") as f:
        documents = pickle.load(f)

    with open(FK_GRAPH_PATH, "rb") as f:
        fk_graph = pickle.load(f)

    return index, documents, fk_graph


def run_preparation():
    """Execute the full Part A preparation pipeline."""
    print("=" * 60)
    print("PART A: Database Preparation")
    print("=" * 60)

    # Step 1: Retrieve complete metadata from MySQL
    print("\n[1/4] Retrieving database metadata from MySQL...")
    metadata = get_complete_metadata()
    schema = metadata["schema"]
    foreign_keys = metadata["foreign_keys"]
    primary_keys = metadata["primary_keys"]
    descriptions = get_table_descriptions()
    print(f"  -> {len(schema)} tables, {len(foreign_keys)} FK relationships, "
          f"{len(primary_keys)} tables with PKs")

    # Step 2: Build enriched table documents
    print("\n[2/4] Building enriched table documents...")
    documents = build_table_documents(schema, primary_keys, foreign_keys, descriptions)
    print(f"  -> {len(documents)} table documents created")
    for doc in documents:
        print(f"     - {doc['table_name']}: {len(doc['columns'])} columns, "
              f"{len(doc['primary_keys'])} PKs, "
              f"{len(doc['foreign_keys_out'])} outgoing FKs")

    # Step 3: Build FAISS index
    print("\n[3/4] Building FAISS vector index...")
    model = SentenceTransformer(MODEL_NAME)
    index, dimension = build_faiss_index(documents, model)
    print(f"  -> Embedding dimension: {dimension}")
    print(f"  -> Vectors in index: {index.ntotal}")

    # Step 4: Build FK relationship graph
    print("\n[4/4] Building foreign-key relationship graph...")
    fk_graph = build_fk_graph(foreign_keys)
    total_edges = sum(len(neighbors) for neighbors in fk_graph.values())
    print(f"  -> {len(fk_graph)} nodes, {total_edges} directed edges (both directions)")

    # Print graph summary
    print("\n  Graph adjacency summary:")
    for table, neighbors in sorted(fk_graph.items()):
        neighbor_names = sorted(neighbors.keys())
        print(f"    {table} -> {', '.join(neighbor_names)}")

    # Persist everything
    print("\n[SAVE] Persisting artifacts to disk...")
    persist_artifacts(index, documents, fk_graph)
    print(f"  -> FAISS index:      {FAISS_INDEX_PATH}")
    print(f"  -> Schema documents: {SCHEMA_DOCS_PATH}")
    print(f"  -> FK graph:         {FK_GRAPH_PATH}")

    print("\n" + "=" * 60)
    print("Part A preparation complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_preparation()
