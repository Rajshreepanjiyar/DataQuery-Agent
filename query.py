"""
query.py — CLI Interface for the NL-to-SQL Agent

Interactive command-line interface that accepts natural language questions,
runs them through the LangGraph agent workflow, and displays the generated SQL
along with actual database results.
"""

import json
import pandas as pd
from graph import agent_graph


def format_output(result: dict):
    """Pretty-print the agent's final response to the console."""
    final = result.get("final_response", {})
    sql = final.get("sql", "")
    message = final.get("message", "")
    error = final.get("error", "")
    results_table = final.get("results_table")

    print("\n" + "=" * 70)

    # Show SQL
    if sql:
        print("\nGENERATED SQL:")
        print("-" * 50)
        print(sql)
        print("-" * 50)

    # Show error if any
    if error:
        print(f"\nERROR: {error}")

    # Show message
    if message:
        print(f"\n{message}")

    # Show results table
    if results_table:
        print("\nRESULTS:")
        print("-" * 50)
        try:
            df = pd.DataFrame(results_table)
            with pd.option_context(
                "display.max_rows", 50,
                "display.max_columns", 20,
                "display.width", 200,
                "display.max_colwidth", 40
            ):
                print(df.to_string(index=False))
        except Exception:
            # Fallback: print as-is
            for row in results_table[:20]:
                print(row)
        print("-" * 50)

    print("=" * 70 + "\n")


def run_query(question: str) -> dict:
    """Run a single question through the LangGraph agent and return the result."""
    initial_state = {
        "user_question": question,
        "sql_correction_count": 0,
        "validation_error_history": [],
    }
    result = agent_graph.invoke(initial_state)
    return result


def main():
    """Interactive CLI loop."""
    print("=" * 70)
    print("  Natural Language to SQL Agent (LangGraph)")
    print("  Ask questions about the Sakila film rental database.")
    print("  Type 'exit', 'quit', or 'bye' to stop.")
    print("=" * 70 + "\n")

    while True:
        try:
            user_q = input("Ask your question: ").strip()
            if not user_q:
                continue
            if user_q.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye!")
                break
            result = run_query(user_q)
            format_output(result)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
