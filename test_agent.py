"""
test_agent.py — Quick integration test for the NL-to-SQL LangGraph agent.

Tests single-table, multi-table (direct JOIN), multi-table (indirect JOIN),
SQL validation (safe and unsafe), and irrelevant question handling.
"""

import json
import sys


def print_result(label, result):
    """Pretty-print a single test result."""
    final = result.get("final_response", {})
    print(f"\n{'='*70}")
    print(f"TEST: {label}")
    print(f"{'='*70}")
    print(f"SQL: {final.get('sql', '(none)')}")
    print(f"Message: {final.get('message', '')}")
    print(f"Error: {final.get('error', '')}")
    results = final.get("results_table")
    if results:
        print(f"Rows returned: {len(results)}")
        # Show first 3 rows
        for i, row in enumerate(results[:3]):
            print(f"  Row {i+1}: {row}")
        if len(results) > 3:
            print(f"  ... ({len(results)} total)")
    print()


def main():
    # Import here so we can catch import errors
    from graph import agent_graph

    tests = [
        # 1. Single table query
        ("Single table: List 5 actors", "List 5 actors"),

        # 2. Direct JOIN (two tables)
        ("Direct JOIN: Films in Comedy category", "Show me all films in the Comedy category"),

        # 3. Indirect JOIN (through intermediate tables)
        ("Indirect JOIN: Actors and categories", "Which actors have appeared in Horror films?"),

        # 4. Aggregation
        ("Aggregation: Revenue by store", "What is the total revenue by store?"),

        # 5. Irrelevant question
        ("Irrelevant question", "What is your name?"),
    ]

    for label, question in tests:
        try:
            result = agent_graph.invoke({
                "user_question": question,
                "sql_correction_count": 0,
                "validation_error_history": [],
            })
            print_result(label, result)
        except Exception as e:
            print(f"\n❌ FAILED: {label}")
            print(f"   Error: {e}")
            print()

    # 6. SQL validation test — manually inject unsafe SQL
    print(f"\n{'='*70}")
    print("TEST: SQL Validation (unsafe operation)")
    print(f"{'='*70}")
    from nodes import validate_sql
    unsafe_state = {"generated_sql": "DROP TABLE actor;", "sql_correction_count": 0, "validation_error_history": []}
    val_result = validate_sql(unsafe_state)
    print(f"Validation result: {val_result['validation_result']}")

    safe_state = {"generated_sql": "SELECT * FROM actor LIMIT 5;", "sql_correction_count": 0, "validation_error_history": []}
    val_result2 = validate_sql(safe_state)
    print(f"Safe query result: {val_result2['validation_result']}")
    print()


if __name__ == "__main__":
    main()
