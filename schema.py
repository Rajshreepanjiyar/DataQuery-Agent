"""
schema.py

This module is responsible for dynamically retrieving database metadata from MySQL.
It connects to the MySQL 'sakila' database and queries table structures, column definitions,
primary key constraints, and foreign key relationships using MySQL's INFORMATION_SCHEMA.

Functions:
    get_connection(): Creates and returns a MySQL database connection.
    get_schema(): Returns column details for all tables in the database.
    get_foreign_keys(): Queries INFORMATION_SCHEMA to return all foreign key relationships.
    get_primary_keys(): Queries INFORMATION_SCHEMA to return primary key columns for each table.
    get_complete_metadata(): Combines tables, columns, primary keys, and foreign keys into a single metadata dictionary.
"""


import mysql.connector
def get_connection():
    """Create and return a connection to the Sakila MySQL database."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="sakila"
    )


def get_schema():
    """Retrieve column names, data types, and attributes for all database tables."""
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    schema = {}

    for table in tables:
        table_name = table[0]
        cursor.execute(f"DESCRIBE `{table_name}`")
        columns = cursor.fetchall()
        schema[table_name] = columns

    cursor.close()
    connection.close()
    return schema


def get_foreign_keys():
    """
    Dynamically retrieve foreign-key relationships from MySQL INFORMATION_SCHEMA.
    Returns a list of dicts with child_table, child_column, parent_table, parent_column.
    """
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    query = """
    SELECT 
        TABLE_NAME as child_table,
        COLUMN_NAME as child_column,
        REFERENCED_TABLE_NAME as parent_table,
        REFERENCED_COLUMN_NAME as parent_column
    FROM 
        INFORMATION_SCHEMA.KEY_COLUMN_USAGE
    WHERE 
        TABLE_SCHEMA = 'sakila'
        AND REFERENCED_TABLE_NAME IS NOT NULL
    """
    cursor.execute(query)
    foreign_keys = cursor.fetchall()
    cursor.close()
    connection.close()
    return foreign_keys


def get_primary_keys():
    """
    Dynamically retrieve primary keys for all tables from MySQL INFORMATION_SCHEMA.
    Returns a dictionary mapping table_name -> list of primary key column names.
    """
    connection = get_connection()
    cursor = connection.cursor(dictionary=True)
    query = """
    SELECT 
        TABLE_NAME, 
        COLUMN_NAME
    FROM 
        INFORMATION_SCHEMA.KEY_COLUMN_USAGE
    WHERE 
        TABLE_SCHEMA = 'sakila'
        AND CONSTRAINT_NAME = 'PRIMARY'
    ORDER BY 
        TABLE_NAME, ORDINAL_POSITION
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    connection.close()

    primary_keys = {}
    for row in rows:
        table = row["TABLE_NAME"]
        col = row["COLUMN_NAME"]
        if table not in primary_keys:
            primary_keys[table] = []
        primary_keys[table].append(col)

    return primary_keys


def get_complete_metadata():
    """Return complete schema, primary keys, and foreign keys in a single dictionary."""
    schema = get_schema()
    foreign_keys = get_foreign_keys()
    primary_keys = get_primary_keys()
    return {
        "schema": schema,
        "foreign_keys": foreign_keys,
        "primary_keys": primary_keys
    }


if __name__ == "__main__":
    metadata = get_complete_metadata()
    print("=" * 50)
    print(f"Loaded {len(metadata['schema'])} tables.")
    print(f"Loaded {len(metadata['foreign_keys'])} foreign key relationships.")
    print(f"Loaded primary keys for {len(metadata['primary_keys'])} tables.")
    print("=" * 50)

    print("\nSample Foreign Keys:")
    for fk in metadata["foreign_keys"][:5]:
        print(f"  {fk['child_table']}.{fk['child_column']} -> {fk['parent_table']}.{fk['parent_column']}")
