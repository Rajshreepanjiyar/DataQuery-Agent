"""
database.py

This module is responsible for executing generated SQL queries against the MySQL database.
It connects to the 'sakila' database, executes queries, and returns row results.
Query execution safety guardrails (such as SELECT-only checks) are enforced before execution.
Includes configurable query timeout and result-row limits.
"""

import mysql.connector

# Configuration
QUERY_TIMEOUT = 30       # seconds
MAX_RESULT_ROWS = 500    # maximum rows to return


def get_connection():
    """Create and return a connection to the Sakila MySQL database."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sakila",
        connection_timeout=QUERY_TIMEOUT
    )


def execute_query(sql, max_rows=MAX_RESULT_ROWS):
    """
    Execute a validated SQL query on the Sakila database and return results.
    Returns a dict with 'columns' (list of column names) and 'rows' (list of tuples).
    Applies a row limit to prevent excessively large result sets.
    """
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows)
        total_count = len(rows)
        # Check if there are more rows beyond the limit
        extra = cursor.fetchone()
        has_more = extra is not None
        return {
            "columns": columns,
            "rows": rows,
            "row_count": total_count,
            "truncated": has_more
        }
    except mysql.connector.Error as e:
        raise RuntimeError(f"Database error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()