"""
database.py

This module is responsible for executing generated SQL queries against the MySQL database.
It connects to the 'sakila' database, executes queries, and returns row results.
Query execution safety guardrails (such as SELECT-only checks) are enforced before execution.
"""

import mysql.connector


def execute_query(sql):
    """Execute a valid SQL query on the Sakila database and return all fetched rows."""
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sakila"
    )

    cursor = connection.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result