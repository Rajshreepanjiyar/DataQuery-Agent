"""
llm.py

Groq LLM configuration and utility functions.
The main SQL generation logic is now handled by the LangGraph nodes in nodes.py.
This module provides the shared LLM instance and a standalone generate_sql function
for backward compatibility and direct usage outside the LangGraph workflow.
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

    NOTE: In the LangGraph workflow, SQL generation is handled by nodes.generate_sql().
    This function is retained for standalone usage or backward compatibility.
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

1. RELEVANCE CHECK: If the user's question is conversational, personal (e.g., "what is your name", "who are you", "hello", "how are you"), or irrelevant/unrelated to querying the database, do NOT generate any SQL query. Return EXACTLY:
Please ask a relevant question related to the data.
2. Use ONLY the tables and columns provided in the Final Relevant Schema.
3. Do NOT invent table names.
4. Do NOT invent column names.
5. Do NOT invent foreign-key relationships.
6. Use the provided foreign-key relationships when determining JOIN conditions.
7. The Anchor Tables are starting points, not necessarily the complete set of tables required for the query.
8. You may use intermediate tables from the Final Relevant Schema when they are required to connect tables.
9. Do not use tables that are not present in the Final Relevant Schema.
10. Generate syntactically valid MySQL SQL if the question is asking to query the database.
11. Return ONLY the SQL query, OR the exact message "Please ask a relevant question related to the data." if the question is irrelevant.
12. Do NOT return explanations.
13. Do NOT use markdown code fences.
14. Do NOT write ```sql.
15. If aggregation is required, use appropriate GROUP BY, COUNT, SUM, AVG, MIN, MAX, etc.
16. If filtering is required, use an appropriate WHERE clause.
17. If multiple tables are required, use the provided foreign-key relationships to construct the JOINs.
18. Do not assume relationships that are not provided.
19. Prefer explicit JOIN syntax instead of implicit comma joins.

USER QUESTION:
{question}

ANCHOR TABLES:
{anchor_tables_str}

FOREIGN-KEY RELATIONSHIPS:
{fk_str}

FINAL RELEVANT SCHEMA:
{final_schema}

Generate the response now."""

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
