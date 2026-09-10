"""
llm.py

This module handles communication with the LLM via LangChain / ChatGroq to convert natural language
questions into executable MySQL SQL queries.

It receives structured parameters:
- question: The natural language query from the user.
- anchor_tables: Semantically relevant tables identified via FAISS vector search.
- foreign_key_relationships: String representation of foreign key connections.
- final_schema: Comprehensive schema details (columns, PKs, FKs) for relevant tables.

It constructs the structured prompt and invokes ChatGroq to generate syntactically valid SQL.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Instantiate ChatGroq model (using free, high-performance model)
model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
llm = ChatGroq(
    model=model_name,
    temperature=0
)


def generate_sql(question, anchor_tables, foreign_key_relationships, final_schema):
    """
    Generate MySQL SQL query using ChatGroq based on the user's question,
    identified anchor tables, detected foreign key relationships, and final schema context.
    """
    # Format anchor tables as string if passed as list
    if isinstance(anchor_tables, list):
        anchor_tables_str = "\n".join([f"- {table}" for table in anchor_tables])
    else:
        anchor_tables_str = str(anchor_tables)

    # Format foreign key relationships as string if passed as list
    if isinstance(foreign_key_relationships, list):
        fk_str = "\n".join([f"- {fk}" for fk in foreign_key_relationships])
    else:
        fk_str = str(foreign_key_relationships)

    prompt = f"""You are an expert MySQL SQL query generator.

Your task is to convert the user's natural language question into a valid MySQL SQL query.

You have been provided with:

1. The user's question
2. Anchor tables identified through semantic schema retrieval
3. Foreign-key relationships from the actual database
4. The final relevant schema selected using semantic relevance and database relationships

IMPORTANT RULES:

1. Use ONLY the tables and columns provided in the Final Relevant Schema.
2. Do NOT invent table names.
3. Do NOT invent column names.
4. Do NOT invent foreign-key relationships.
5. Use the provided foreign-key relationships when determining JOIN conditions.
6. The Anchor Tables are starting points, not necessarily the complete set of tables required for the query.
7. You may use intermediate tables from the Final Relevant Schema when they are required to connect tables.
8. Do not use tables that are not present in the Final Relevant Schema.
9. Generate syntactically valid MySQL SQL.
10. Return ONLY the SQL query.
11. Do NOT return explanations.
12. Do NOT use markdown code fences.
13. Do NOT write ```sql.
14. If aggregation is required, use appropriate GROUP BY, COUNT, SUM, AVG, MIN, MAX, etc.
15. If filtering is required, use an appropriate WHERE clause.
16. If multiple tables are required, use the provided foreign-key relationships to construct the JOINs.
17. Do not assume relationships that are not provided.
18. Prefer explicit JOIN syntax instead of implicit comma joins.

USER QUESTION:
{question}

ANCHOR TABLES:
{anchor_tables_str}

FOREIGN-KEY RELATIONSHIPS:
{fk_str}

FINAL RELEVANT SCHEMA:
{final_schema}

Generate the MySQL SQL query now."""

    response = llm.invoke(prompt)
    sql = response.content.strip()

    # Clean markdown formatting if present
    if sql.startswith("```"):
        lines = sql.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        sql = "\n".join(lines).strip()

    return sql