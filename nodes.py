"""
nodes.py — LangGraph Workflow Nodes (Part B)

Each function is a LangGraph node that receives the shared AgentState and returns
a partial state update dict.  Nodes:

  1. understand_question   — LLM-powered question analysis
  2. retrieve_schema       — FAISS similarity search for relevant tables
  3. expand_relationships  — FK graph traversal to find JOIN paths
  4. generate_sql          — LLM-powered SQL generation
  5. validate_sql          — Python-based SQL safety validation
  6. execute_sql           — Run validated SQL against MySQL
  7. format_response       — Build the final user-facing response
"""

import os
import re
import json
from collections import deque

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import faiss

from models import QuestionUnderstanding, SQLValidationResult
from rag import load_artifacts, MODEL_NAME

load_dotenv()

# ==============================================================================
# Shared Resources (loaded once at import time)
# ==============================================================================

# Load persisted artifacts from Part A
_faiss_index, _schema_documents, _fk_graph = load_artifacts()

# Embedding model (same as Part A)
_embed_model = SentenceTransformer(MODEL_NAME)

# Groq LLM
_model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
_llm = ChatGroq(model=_model_name, temperature=0)

# Build a quick lookup: table_name -> document dict
_table_lookup = {doc["table_name"]: doc for doc in _schema_documents}


# ==============================================================================
# Configuration
# ==============================================================================

SIMILARITY_THRESHOLD = 0.15
MAX_FAISS_CANDIDATES = 10
MAX_SQL_CORRECTIONS = 3


# ==============================================================================
# Node 1: Understand Question
# ==============================================================================

def understand_question(state: dict) -> dict:
    """
    Use the Groq LLM to parse the user's natural-language question into a
    structured QuestionUnderstanding (Pydantic model).  Does NOT invent real
    table or column names — only identifies domain concepts.
    """
    question = state["user_question"]

    prompt = f"""You are an expert database analyst. Analyze the following natural language question and produce a JSON object with these fields:

- "intent": The high-level intent (e.g., "retrieve data", "aggregate", "count", "compare", "list").
- "concepts": A list of domain concepts relevant to answering this question (e.g., "customer", "film", "rental", "payment", "actor", "category"). Do NOT use actual database table or column names — use plain English concepts.
- "requested_output": What the user wants returned (e.g., "list of names", "total count", "average revenue").
- "filters": A list of any filter conditions mentioned (e.g., "active customers", "released in 2006", "rating PG-13"). Empty list if none.
- "grouping": Grouping dimensions if any (e.g., "by category", "per store"). Empty list if none.
- "sorting": Sorting requirement if any (e.g., "highest first", "alphabetically"). Empty string if none.
- "aggregations": Aggregation operations needed (e.g., "COUNT", "SUM", "AVG"). Empty list if none.
- "is_relevant": true if the question is about querying or analyzing data from a film rental database, false if it is conversational, personal, or irrelevant (e.g., "hello", "who are you", "what is the weather").

Return ONLY valid JSON. No explanations, no markdown fences.

Question: {question}"""

    response = _llm.invoke(prompt)
    raw = response.content.strip()

    # Clean markdown if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
        understanding = QuestionUnderstanding(**parsed)
    except Exception:
        # Fallback: treat as relevant with minimal structure
        understanding = QuestionUnderstanding(
            intent="retrieve data",
            concepts=[question],
            requested_output="data",
            is_relevant=True
        )

    return {
        "question_understanding": understanding.model_dump()
    }


# ==============================================================================
# Node 2: Retrieve Schema
# ==============================================================================

def retrieve_schema(state: dict) -> dict:
    """
    Embed the user question with SentenceTransformers and search the FAISS index
    to retrieve the most relevant tables.  Uses both semantic similarity scoring
    and a keyword boost for exact table-name mentions.
    """
    question = state["user_question"]
    understanding = state.get("question_understanding", {})
    concepts = understanding.get("concepts", [])

    # Build a combined search text from the question and extracted concepts
    search_text = question
    if concepts:
        search_text += " " + " ".join(concepts)

    # Embed and search FAISS
    query_embedding = _embed_model.encode([search_text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    scores, indices = _faiss_index.search(query_embedding, k=MAX_FAISS_CANDIDATES)

    anchor_tables = []
    relevant_schemas = []
    question_lower = question.lower()

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_schema_documents):
            continue
        doc = _schema_documents[idx]
        table_name = doc["table_name"]
        sim_score = float(score)

        # Keyword boost: if the table name appears literally in the question
        if table_name.replace("_", " ") in question_lower or table_name in question_lower:
            sim_score = max(sim_score, SIMILARITY_THRESHOLD + 0.1)

        if sim_score >= SIMILARITY_THRESHOLD:
            anchor_tables.append({
                "table_name": table_name,
                "similarity_score": round(sim_score, 4)
            })
            relevant_schemas.append(doc)

    # Fallback: always include at least the top-1 result
    if not anchor_tables and len(indices[0]) > 0:
        top_idx = int(indices[0][0])
        if 0 <= top_idx < len(_schema_documents):
            doc = _schema_documents[top_idx]
            anchor_tables.append({
                "table_name": doc["table_name"],
                "similarity_score": round(float(scores[0][0]), 4)
            })
            relevant_schemas.append(doc)

    return {
        "anchor_tables": anchor_tables,
        "relevant_schemas": relevant_schemas
    }


# ==============================================================================
# Node 3: Expand Relationships
# ==============================================================================

def _bfs_shortest_path(graph, start, end):
    """BFS to find the shortest path between two tables in the FK graph."""
    if start == end:
        return [start]
    if start not in graph:
        return []

    visited = {start}
    queue = deque([[start]])

    while queue:
        path = queue.popleft()
        current = path[-1]

        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                if neighbor == end:
                    return new_path
                queue.append(new_path)
    return []


def _get_join_info(graph, table_a, table_b):
    """Get the FK column mappings between two directly connected tables."""
    edges = graph.get(table_a, {}).get(table_b, [])
    return edges


def expand_relationships(state: dict) -> dict:
    """
    Use the FK graph to:
      1. Find shortest paths between all pairs of anchor tables.
      2. Include intermediate tables needed for indirect JOINs.
      3. Collect exact FK column mappings for every edge in the paths.
      4. Build a formatted schema context string for the LLM.
    """
    anchor_tables = [t["table_name"] for t in state.get("anchor_tables", [])]
    question = state["user_question"]
    understanding = state.get("question_understanding", {})

    if not anchor_tables:
        return {
            "expanded_tables": [],
            "join_relationships": [],
            "schema_context": "No relevant tables found."
        }

    # Discover all tables needed (anchors + intermediates)
    all_tables = set(anchor_tables)
    path_edges = []  # (table_a, table_b) pairs along discovered paths

    if len(anchor_tables) == 1:
        # For a single anchor, include its direct FK neighbors
        single = anchor_tables[0]
        neighbors = list(_fk_graph.get(single, {}).keys())
        all_tables.update(neighbors)
        for neighbor in neighbors:
            path_edges.append((single, neighbor))
    else:
        # For multiple anchors, find shortest paths between all pairs
        for i in range(len(anchor_tables)):
            for j in range(i + 1, len(anchor_tables)):
                path = _bfs_shortest_path(_fk_graph, anchor_tables[i], anchor_tables[j])
                if path:
                    all_tables.update(path)
                    for k in range(len(path) - 1):
                        path_edges.append((path[k], path[k + 1]))

    # Collect JOIN relationship details for every edge
    join_relationships = []
    seen_joins = set()
    for table_a, table_b in path_edges:
        edges = _get_join_info(_fk_graph, table_a, table_b)
        for edge in edges:
            # Create a canonical key to avoid duplicate joins
            join_key = (
                edge["child_table"], edge["child_column"],
                edge["parent_table"], edge["parent_column"]
            )
            if join_key not in seen_joins:
                seen_joins.add(join_key)
                join_relationships.append({
                    "child_table": edge["child_table"],
                    "child_column": edge["child_column"],
                    "parent_table": edge["parent_table"],
                    "parent_column": edge["parent_column"],
                    "join_condition": (
                        f"{edge['child_table']}.{edge['child_column']} = "
                        f"{edge['parent_table']}.{edge['parent_column']}"
                    )
                })

    # Build formatted schema context for the LLM
    expanded_tables = sorted(all_tables)
    schema_context = _build_schema_context(expanded_tables, join_relationships)

    return {
        "expanded_tables": expanded_tables,
        "join_relationships": join_relationships,
        "schema_context": schema_context
    }


def _build_schema_context(tables, join_relationships):
    """Format the schema context string for the SQL generation LLM prompt."""
    blocks = []

    for table_name in tables:
        doc = _table_lookup.get(table_name)
        if not doc:
            continue

        block_lines = [f"TABLE: {table_name}", "", "COLUMNS:"]
        for col in doc["columns"]:
            line = f"  {col['name']} {col['type']}"
            if col.get("key") == "PRI":
                line += " PRIMARY KEY"
            if col.get("extra"):
                line += f" {col['extra']}"
            block_lines.append(line)

        if doc["primary_keys"]:
            block_lines.append(f"\nPRIMARY KEY: {', '.join(doc['primary_keys'])}")

        # Outgoing FKs relevant to our table set
        relevant_out = [
            fk for fk in doc["foreign_keys_out"]
            if fk["ref_table"] in tables
        ]
        if relevant_out:
            block_lines.append("\nFOREIGN KEYS:")
            for fk in relevant_out:
                block_lines.append(
                    f"  {table_name}.{fk['column']} -> {fk['ref_table']}.{fk['ref_column']}"
                )

        blocks.append("\n".join(block_lines))

    # Append JOIN relationships summary
    if join_relationships:
        join_lines = ["\n\nJOIN RELATIONSHIPS:"]
        for jr in join_relationships:
            join_lines.append(f"  {jr['join_condition']}")
        blocks.append("\n".join(join_lines))

    return "\n\n".join(blocks)


# ==============================================================================
# Node 4: Generate SQL
# ==============================================================================

def generate_sql(state: dict) -> dict:
    """
    Pass the question, structured understanding, schema context, and join
    relationships to the Groq LLM to generate a MySQL-compatible SELECT query.
    """
    question = state["user_question"]
    understanding = state.get("question_understanding", {})
    schema_context = state.get("schema_context", "")
    join_relationships = state.get("join_relationships", [])
    expanded_tables = state.get("expanded_tables", [])
    correction_count = state.get("sql_correction_count", 0)
    error_history = state.get("validation_error_history", [])

    # Check if question is relevant
    if not understanding.get("is_relevant", True):
        return {
            "generated_sql": "",
            "sql_correction_count": correction_count
        }

    # Format join relationships for the prompt
    join_info = ""
    if join_relationships:
        join_info = "\n".join([f"  - {jr['join_condition']}" for jr in join_relationships])
    else:
        join_info = "  (No explicit join relationships identified)"

    # Format anchor tables
    anchor_str = "\n".join([f"  - {t}" for t in expanded_tables])

    # Build correction context if this is a retry
    correction_context = ""
    if error_history:
        correction_context = f"""

IMPORTANT — PREVIOUS ATTEMPT FAILED VALIDATION:
The previously generated SQL was rejected. Errors:
{chr(10).join(f'  - {err}' for err in error_history)}

Please correct these issues and generate a valid SQL query.
"""

    prompt = f"""You are an expert MySQL SQL query generator for a film rental database (Sakila).

Generate a valid MySQL SELECT query to answer the user's question.

STRICT RULES:
1. Use ONLY the tables and columns provided in the schema below.
2. Do NOT invent table names, column names, or relationships.
3. Generate ONLY a single SELECT statement (CTEs with WITH ... AS ... SELECT are allowed).
4. Do NOT generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any data-modification statements.
5. Use explicit JOIN syntax with the provided foreign-key relationships.
6. Use appropriate WHERE, GROUP BY, HAVING, ORDER BY, and LIMIT clauses as needed.
7. Return ONLY the raw SQL query. No explanations, no markdown fences, no comments.
8. If the schema is insufficient to answer the question, return exactly: INSUFFICIENT_SCHEMA
9. If aggregation is required, use appropriate GROUP BY with COUNT, SUM, AVG, MIN, MAX.

USER QUESTION:
{question}

QUESTION ANALYSIS:
  Intent: {understanding.get('intent', 'unknown')}
  Requested Output: {understanding.get('requested_output', 'data')}
  Filters: {', '.join(understanding.get('filters', [])) or 'none'}
  Grouping: {', '.join(understanding.get('grouping', [])) or 'none'}
  Sorting: {understanding.get('sorting', 'none')}
  Aggregations: {', '.join(understanding.get('aggregations', [])) or 'none'}

AVAILABLE TABLES:
{anchor_str}

JOIN RELATIONSHIPS:
{join_info}

DATABASE SCHEMA:
{schema_context}
{correction_context}
Generate the SQL query now."""

    response = _llm.invoke(prompt)
    sql = response.content.strip()

    # Clean markdown fences if present
    if sql.startswith("```"):
        lines = sql.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        sql = "\n".join(lines).strip()

    return {
        "generated_sql": sql,
        "sql_correction_count": correction_count
    }


# ==============================================================================
# Node 5: Validate SQL
# ==============================================================================

# Prohibited SQL keywords for write/modify operations
_PROHIBITED_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "RENAME", "GRANT", "REVOKE",
    "LOAD", "CALL", "EXEC", "EXECUTE",
    "SET", "LOCK", "UNLOCK", "FLUSH", "RESET", "PURGE",
    "INTO OUTFILE", "INTO DUMPFILE",
]


def validate_sql(state: dict) -> dict:
    """
    Validate the generated SQL:
      - Must not be empty
      - Must be a single statement
      - Must start with SELECT or WITH (for CTEs)
      - Must not contain prohibited modification keywords
      - Basic syntax checks
    """
    sql = state.get("generated_sql", "").strip()
    correction_count = state.get("sql_correction_count", 0)
    error_history = state.get("validation_error_history", [])

    # Handle non-SQL responses (e.g., INSUFFICIENT_SCHEMA or irrelevant question)
    if not sql or sql == "INSUFFICIENT_SCHEMA":
        result = SQLValidationResult(
            is_valid=False,
            error_message="No SQL query was generated. The schema may be insufficient for this question.",
            error_type="empty"
        )
        return {
            "validation_result": result.model_dump(),
            "sql_correction_count": correction_count,
            "validation_error_history": error_history
        }

    # Check for multiple statements (semicolon-separated)
    # Remove semicolons inside string literals before checking
    sql_no_strings = re.sub(r"'[^']*'", "", sql)
    sql_no_strings = re.sub(r'"[^"]*"', "", sql_no_strings)
    statements = [s.strip() for s in sql_no_strings.split(";") if s.strip()]
    if len(statements) > 1:
        error_msg = "Multiple SQL statements detected. Only a single SELECT statement is allowed."
        result = SQLValidationResult(
            is_valid=False,
            error_message=error_msg,
            error_type="multiple_statements"
        )
        return {
            "validation_result": result.model_dump(),
            "sql_correction_count": correction_count + 1,
            "validation_error_history": error_history + [error_msg]
        }

    # Check that the query starts with SELECT or WITH (CTE)
    sql_upper = sql.upper().lstrip()
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        error_msg = f"Query must start with SELECT or WITH (CTE). Found: {sql_upper[:30]}..."
        result = SQLValidationResult(
            is_valid=False,
            error_message=error_msg,
            error_type="unsafe_operation"
        )
        return {
            "validation_result": result.model_dump(),
            "sql_correction_count": correction_count + 1,
            "validation_error_history": error_history + [error_msg]
        }

    # Check for prohibited keywords
    # Use word-boundary matching to avoid false positives (e.g., "UPDATED_AT" column)
    for keyword in _PROHIBITED_KEYWORDS:
        pattern = r"\b" + keyword.replace(" ", r"\s+") + r"\b"
        if re.search(pattern, sql_no_strings, re.IGNORECASE):
            # Special case: "SET" can appear in legitimate contexts like CHARSET
            if keyword == "SET" and re.search(r"\bCHARACTER\s+SET\b", sql_no_strings, re.IGNORECASE):
                continue
            error_msg = f"Prohibited SQL operation detected: {keyword}. Only read-only SELECT queries are allowed."
            result = SQLValidationResult(
                is_valid=False,
                error_message=error_msg,
                error_type="unsafe_operation"
            )
            return {
                "validation_result": result.model_dump(),
                "sql_correction_count": correction_count + 1,
                "validation_error_history": error_history + [error_msg]
            }

    # Basic bracket/parenthesis balance check
    open_parens = sql.count("(")
    close_parens = sql.count(")")
    if open_parens != close_parens:
        error_msg = f"Unbalanced parentheses: {open_parens} opening vs {close_parens} closing."
        result = SQLValidationResult(
            is_valid=False,
            error_message=error_msg,
            error_type="syntax"
        )
        return {
            "validation_result": result.model_dump(),
            "sql_correction_count": correction_count + 1,
            "validation_error_history": error_history + [error_msg]
        }

    # All checks passed
    result = SQLValidationResult(is_valid=True)
    return {
        "validation_result": result.model_dump(),
        "sql_correction_count": correction_count,
        "validation_error_history": error_history
    }


# ==============================================================================
# Node 6: Execute SQL
# ==============================================================================

def execute_sql(state: dict) -> dict:
    """Execute the validated SQL query against the Sakila database."""
    from database import execute_query

    sql = state.get("generated_sql", "").strip()
    validation = state.get("validation_result", {})

    if not validation.get("is_valid", False):
        return {
            "query_results": None,
            "execution_error": "Cannot execute: SQL failed validation."
        }

    try:
        results = execute_query(sql)
        return {
            "query_results": results,
            "execution_error": ""
        }
    except Exception as e:
        return {
            "query_results": None,
            "execution_error": str(e)
        }


# ==============================================================================
# Node 7: Format Response
# ==============================================================================

def format_response(state: dict) -> dict:
    """
    Build the final user-facing response containing:
      - The generated SQL query
      - Query results in a clean format
      - Error messages when applicable
    """
    sql = state.get("generated_sql", "")
    understanding = state.get("question_understanding", {})
    results = state.get("query_results")
    execution_error = state.get("execution_error", "")
    validation = state.get("validation_result", {})

    response = {
        "sql": sql,
        "results_table": None,
        "message": "",
        "error": ""
    }

    # Handle irrelevant questions
    if not understanding.get("is_relevant", True):
        response["message"] = "Please ask a relevant question related to the data."
        response["sql"] = ""
        return {"final_response": response}

    # Handle insufficient schema
    if sql == "INSUFFICIENT_SCHEMA" or not sql:
        response["message"] = "The available schema is insufficient to answer this question."
        response["sql"] = ""
        return {"final_response": response}

    # Handle validation failure (after max retries exhausted)
    if not validation.get("is_valid", False):
        response["error"] = f"SQL validation failed: {validation.get('error_message', 'Unknown error')}"
        return {"final_response": response}

    # Handle execution error
    if execution_error:
        response["error"] = f"SQL execution error: {execution_error}"
        return {"final_response": response}

    # Handle successful results
    if results:
        columns = results.get("columns", [])
        rows = results.get("rows", [])
        row_count = results.get("row_count", 0)
        truncated = results.get("truncated", False)

        if rows:
            # Convert rows to list of dicts for clean display
            results_table = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    val = row[i] if i < len(row) else None
                    # Convert non-serializable types to strings
                    if isinstance(val, (bytes, bytearray)):
                        val = val.decode("utf-8", errors="replace")
                    elif hasattr(val, "isoformat"):
                        val = val.isoformat()
                    elif isinstance(val, __import__("decimal").Decimal):
                        val = float(val)
                    row_dict[col] = val
                results_table.append(row_dict)

            response["results_table"] = results_table
            message = f"Found {row_count} result(s)."
            if truncated:
                message += " (Results truncated — more rows available in the database.)"
            response["message"] = message
        else:
            response["message"] = "Query executed successfully but returned no results."
    else:
        response["message"] = "Query executed but no results were returned."

    return {"final_response": response}


# ==============================================================================
# Conditional Edge: Should Retry SQL?
# ==============================================================================

def should_retry_sql(state: dict) -> str:
    """
    Conditional router after validate_sql:
      - "retry"    → route back to generate_sql (if validation failed and retries remain)
      - "execute"  → route to execute_sql (if validation passed)
      - "fail"     → route to format_response (if max retries exhausted)
    """
    validation = state.get("validation_result", {})
    correction_count = state.get("sql_correction_count", 0)

    if validation.get("is_valid", False):
        return "execute"

    # Check if the error is non-retryable (empty/insufficient schema)
    if validation.get("error_type") == "empty":
        return "fail"

    if correction_count < MAX_SQL_CORRECTIONS:
        return "retry"

    return "fail"
