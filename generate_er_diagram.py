import mysql.connector
from graphviz import Digraph


# Connect to MySQL
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="sakila"
)

cursor = connection.cursor(dictionary=True)


# Get all tables
cursor.execute("""
    SELECT TABLE_NAME
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = 'sakila'
    AND TABLE_TYPE = 'BASE TABLE'
    ORDER BY TABLE_NAME
""")

tables = cursor.fetchall()


# Get primary keys
cursor.execute("""
    SELECT
        TABLE_NAME,
        COLUMN_NAME
    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
    WHERE TABLE_SCHEMA = 'sakila'
    AND CONSTRAINT_NAME = 'PRIMARY'
""")

primary_keys = {}

for row in cursor.fetchall():
    primary_keys.setdefault(row["TABLE_NAME"], set()).add(row["COLUMN_NAME"])


# Get foreign keys
cursor.execute("""
    SELECT
        TABLE_NAME,
        COLUMN_NAME,
        REFERENCED_TABLE_NAME,
        REFERENCED_COLUMN_NAME
    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
    WHERE TABLE_SCHEMA = 'sakila'
    AND REFERENCED_TABLE_NAME IS NOT NULL
""")

foreign_keys = cursor.fetchall()


# Get columns
cursor.execute("""
    SELECT
        TABLE_NAME,
        COLUMN_NAME,
        COLUMN_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'sakila'
    ORDER BY TABLE_NAME, ORDINAL_POSITION
""")

columns = cursor.fetchall()


# Organize columns by table
table_columns = {}

for column in columns:
    table_columns.setdefault(column["TABLE_NAME"], []).append(column)


# Create Graphviz diagram
dot = Digraph(
    "Sakila_ER_Diagram",
    format="png"
)

dot.attr(
    rankdir="LR",
    bgcolor="white",
    splines="ortho",
    nodesep="0.5",
    ranksep="1"
)


# Create table boxes
for table in tables:

    table_name = table["TABLE_NAME"]

    label = f'<<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0">'

    # Table name
    label += f'''
    <TR>
        <TD COLSPAN="2" BGCOLOR="lightblue">
            <B>{table_name}</B>
        </TD>
    </TR>
    '''

    # Columns
    for column in table_columns.get(table_name, []):

        column_name = column["COLUMN_NAME"]
        column_type = column["COLUMN_TYPE"]

        # Identify PK
        if column_name in primary_keys.get(table_name, set()):
            key_symbol = "🔑 PK"
        else:
            key_symbol = ""

        # Identify FK
        is_fk = any(
            fk["TABLE_NAME"] == table_name
            and fk["COLUMN_NAME"] == column_name
            for fk in foreign_keys
        )

        if is_fk:
            key_symbol = "🔗 FK" if not key_symbol else "🔑 PK / FK"

        label += f'''
        <TR>
            <TD ALIGN="LEFT">
                {key_symbol} {column_name}
            </TD>
            <TD ALIGN="LEFT">
                {column_type}
            </TD>
        </TR>
        '''

    label += "</TABLE>>"

    dot.node(
        table_name,
        label=label,
        shape="plain"
    )


# Create relationships
for fk in foreign_keys:

    child_table = fk["TABLE_NAME"]
    child_column = fk["COLUMN_NAME"]

    parent_table = fk["REFERENCED_TABLE_NAME"]
    parent_column = fk["REFERENCED_COLUMN_NAME"]

    dot.edge(
        child_table,
        parent_table,
        label=f"{child_column} → {parent_column}",
        dir="forward"
    )


# Generate PNG
output_file = dot.render(
    filename="sakila_er_diagram",
    cleanup=True
)

print("ER diagram generated successfully!")
print("File:", output_file)


cursor.close()
connection.close()