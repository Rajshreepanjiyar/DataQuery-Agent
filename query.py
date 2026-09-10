"""
query.py

This module implements the runtime retrieval and SQL generation pipeline:
1. Accepts natural language user questions.
2. Generates question embeddings using SentenceTransformer ('all-MiniLM-L6-v2').
3. Searches FAISS index to identify semantically relevant 'Anchor Tables' using a configurable distance threshold.
4. Dynamically retrieves foreign key relationships and primary keys from MySQL via schema.py.
5. Performs BFS graph traversal to discover intermediate connecting tables between anchor tables.
6. Builds a detailed Final Relevant Schema context (table name, column names, data types, PKs, FKs).
7. Passes Question, Anchor Tables, FK Relationships, and Final Schema to llm.py to generate SQL.
8. Executes the generated SQL query in MySQL via database.py with SELECT safety verification.
9. Displays comprehensive step-by-step diagnostic logging output.
"""

from collections import deque
import pickle
import faiss
from sentence_transformers import SentenceTransformer

from schema import get_complete_metadata
from llm import generate_sql
from database import execute_query

# ==================================================
# CONFIGURATION
# ==================================================
# Cosine Similarity Threshold for Anchor Table selection (0.0 to 1.0).
# Higher score = higher semantic similarity.
# Note: Can be calibrated using benchmark queries based on vector similarity distributions.
SIMILARITY_THRESHOLD = 0.15
MAX_CANDIDATES = 10

# Load FAISS vector index and schema documents metadata
index = faiss.read_index("schema.index")
with open("schema_documents.pkl", "rb") as file:
    documents = pickle.load(file)

# Load SentenceTransformer embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def identify_anchor_tables(question, max_candidates=MAX_CANDIDATES, threshold=SIMILARITY_THRESHOLD):
    """
    Search FAISS index using question embedding to identify semantically relevant anchor tables.
    Uses Cosine Similarity score thresholding rather than blindly taking top-k candidates.
    """
    query_embedding = model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, k=max_candidates)

    anchor_tables = []
    anchor_details = []

    for score, idx in zip(scores[0], indices[0]):
        table_name = documents[idx]["table_name"]
        if score >= threshold:
            anchor_tables.append(table_name)
            anchor_details.append((table_name, float(score)))

    # Fallback: If no candidate satisfies the threshold, select top-1 table as anchor
    if not anchor_tables and len(indices[0]) > 0:
        top_idx = indices[0][0]
        top_score = float(scores[0][0])
        top_table = documents[top_idx]["table_name"]
        anchor_tables.append(top_table)
        anchor_details.append((top_table, top_score))

    return anchor_tables, anchor_details



def build_fk_graph(foreign_keys):
    """
    Build an undirected graph adjacency list representation of database tables
    connected by foreign key relationships.
    """
    graph = {}
    for fk in foreign_keys:
        child = fk["child_table"]
        parent = fk["parent_table"]

        if child not in graph:
            graph[child] = set()
        if parent not in graph:
            graph[parent] = set()

        graph[child].add(parent)
        graph[parent].add(child)

    return graph


def bfs_shortest_path(graph, start_table, end_table):
    """Find the shortest table path between start_table and end_table using BFS."""
    if start_table == end_table:
        return [start_table]

    visited = {start_table}
    queue = deque([[start_table]])

    while queue:
        path = queue.popleft()
        current_node = path[-1]

        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path) + [neighbor]
                if neighbor == end_table:
                    return new_path
                queue.append(new_path)

    return []


def discover_connecting_tables(anchor_tables, graph):
    """
    Analyze foreign key graph to discover intermediate tables connecting anchor tables.
    Avoids blindly adding the entire database.
    """
    if not anchor_tables:
        return set()

    # If only 1 anchor table, include anchor table and its immediate FK neighbors
    if len(anchor_tables) == 1:
        single_anchor = anchor_tables[0]
        neighbors = set(graph.get(single_anchor, []))
        return {single_anchor}.union(neighbors)

    # For multiple anchor tables, find shortest connecting paths between all anchor pairs
    final_tables = set(anchor_tables)
    for i in range(len(anchor_tables)):
        for j in range(i + 1, len(anchor_tables)):
            path = bfs_shortest_path(graph, anchor_tables[i], anchor_tables[j])
            if path:
                final_tables.update(path)

    return final_tables


def construct_final_schema_context(relevant_tables, schema, primary_keys, foreign_keys):
    """
    Construct structured schema context containing table columns, data types,
    primary keys, and foreign keys for all relevant tables.
    """
    schema_blocks = []
    relevant_fk_strings = []

    # Gather formatted foreign keys relevant to the selected tables
    for fk in foreign_keys:
        if fk["child_table"] in relevant_tables and fk["parent_table"] in relevant_tables:
            fk_str = f"{fk['child_table']}.{fk['child_column']} -> {fk['parent_table']}.{fk['parent_column']}"
            if fk_str not in relevant_fk_strings:
                relevant_fk_strings.append(fk_str)

    # Build schema block for each relevant table
    for table_name in sorted(relevant_tables):
        if table_name not in schema:
            continue

        columns = schema[table_name]
        block = f"TABLE: {table_name}\n\nCOLUMNS:\n"

        for col in columns:
            col_name = col[0]
            data_type = col[1]
            block += f"{col_name} - {data_type}\n"

        # Primary Keys
        pks = primary_keys.get(table_name, [])
        if pks:
            if len(pks) == 1:
                pk_str = f"{table_name}.{pks[0]}"
            else:
                pk_str = f"({', '.join(pks)})"
            block += f"\nPRIMARY KEY:\n{pk_str}\n"

        # Foreign Keys for this table
        table_fks = [
            f"{fk['child_table']}.{fk['child_column']} -> {fk['parent_table']}.{fk['parent_column']}"
            for fk in foreign_keys if fk["child_table"] == table_name and fk["parent_table"] in relevant_tables
        ]
        if table_fks:
            block += f"\nFOREIGN KEYS:\n" + "\n".join(table_fks) + "\n"

        schema_blocks.append(block)

    final_schema_text = "\n\n".join(schema_blocks)
    return final_schema_text, relevant_fk_strings


def run_pipeline(user_question):
    """Execute the complete NL-to-SQL pipeline with step-by-step diagnostic logging."""
    print("\n" + "=" * 60)
    print("USER QUESTION:")
    print(user_question)
    print("=" * 60)

    # Step 1: Anchor Table Retrieval via FAISS
    anchor_tables, anchor_details = identify_anchor_tables(user_question)

    print("\nANCHOR TABLES:")
    if anchor_details:
        for t, score in anchor_details:
            print(f"- {t} (similarity score: {score:.4f})")
    else:
        print("- None")

    # Step 2 & 3: Foreign Key Graph Retrieval & Intermediate Table Discovery
    db_metadata = get_complete_metadata()
    schema_data = db_metadata["schema"]
    foreign_keys = db_metadata["foreign_keys"]
    primary_keys = db_metadata["primary_keys"]

    fk_graph = build_fk_graph(foreign_keys)
    final_relevant_tables = discover_connecting_tables(anchor_tables, fk_graph)

    # Step 4: Final Schema Context Construction
    final_schema_text, relevant_fk_strings = construct_final_schema_context(
        final_relevant_tables, schema_data, primary_keys, foreign_keys
    )

    print("\nDETECTED FK RELATIONSHIPS:")
    if relevant_fk_strings:
        for fk_s in relevant_fk_strings:
            print(f"- {fk_s}")
    else:
        print("- None")

    print("\nFINAL RELEVANT TABLES:")
    for t in sorted(final_relevant_tables):
        print(f"- {t}")

    print("\nFINAL SCHEMA CONTEXT:")
    print("-" * 50)
    print(final_schema_text)
    print("-" * 50)

    # Step 5: LLM SQL Generation
    sql_query = generate_sql(
        question=user_question,
        anchor_tables=anchor_tables,
        foreign_key_relationships=relevant_fk_strings,
        final_schema=final_schema_text
    )

    print("\nGENERATED SQL:")
    print(sql_query)

    # Step 6: SQL Execution with SELECT-only safety check
    print("\nDATABASE RESULT:")
    try:
        if not sql_query.strip().lower().startswith("select"):
            raise ValueError("Execution blocked: Only SELECT queries are permitted.")

        results = execute_query(sql_query)
        if results:
            for row in results[:20]:  # Display top 20 rows
                print(row)
            if len(results) > 20:
                print(f"... ({len(results)} rows returned in total)")
        else:
            print("(0 rows returned)")
    except Exception as e:
        print("\nError executing SQL:")
        print(e)

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("=" * 60)
    print("SQL Agent is ready! Type 'exit', 'quit', or 'bye' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_q = input("Ask your question: ").strip()
            if not user_q:
                continue
            if user_q.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye!")
                break
            run_pipeline(user_q)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

