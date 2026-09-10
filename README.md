# SchemaQuery — Natural Language-to-SQL Agent

SchemaQuery is an AI-powered Natural Language-to-SQL Agent that allows users to ask questions about a MySQL database using simple natural language instead of writing SQL queries manually.

The project combines **LLMs, semantic search, vector embeddings, FAISS, and database relationship graphs** to identify the relevant database schema and generate SQL queries.

---

## 📌 Problem Statement

Working with relational databases often requires users to know SQL and understand the database structure.

For example, to answer a simple question such as:

> "What are the top 5 customers who have made the highest total payments?"

a user needs to know:

- Which tables contain customer and payment information
- Which columns should be used
- How the tables are related
- How to write the required JOIN
- How to use aggregation and sorting in SQL

This becomes more difficult when a database contains many tables and relationships.

Another challenge is providing the **entire database schema to an LLM for every question**. Large schemas can increase the amount of context sent to the model and may make it harder for the LLM to identify the correct tables and relationships.

---

## 💡 Solution

SchemaQuery solves this problem by creating a retrieval-based pipeline that identifies only the relevant part of the database before asking the LLM to generate SQL.

Instead of directly sending the complete database schema to the LLM, the system:

1. Converts the database schema into embeddings.
2. Stores those embeddings in a FAISS vector index.
3. Converts the user's question into an embedding.
4. Finds semantically relevant tables using similarity search.
5. Selects relevant tables as **anchor tables** using a similarity threshold.
6. Builds a graph from the database's actual foreign-key relationships.
7. Uses BFS to find connecting paths between anchor tables.
8. Creates a final schema containing the relevant tables, columns, data types, primary keys, and foreign keys.
9. Sends this focused schema context to the LLM.
10. Generates a SQL query.
11. Performs a basic read-only safety check.
12. Executes the query in MySQL and displays the results.

---

## 🔄 Project Workflow

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Similarity Search
      ↓
Anchor Tables
      ↓
Foreign-Key Graph
      ↓
BFS Shortest-Path Search
      ↓
Relevant Tables
      ↓
Final Schema Context
      ↓
LLM
      ↓
Generated SQL
      ↓
SELECT Safety Check
      ↓
MySQL Database
      ↓
Query Results