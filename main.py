"""
main.py — Entry Point for the NL-to-SQL Data Query Agent

This is the primary entry point for running the agent. It accepts a natural
language question, runs the complete LangGraph workflow, and displays both
the generated SQL query and actual database results.

Usage:
    python main.py                      # Interactive mode (loop)
    python main.py "your question"      # Single question mode
"""

import sys
import pandas as pd
from graph import agent_graph


def run_query(question: str) -> dict:
    """Run a single question through the LangGraph agent and return the full state."""
    initial_state = {
        "user_question": question,
        "sql_correction_count": 0,
        "validation_error_history": [],
    }
    result = agent_graph.invoke(initial_state)
    return result


def display_result(result: dict):
    """Pretty-print the agent's final response to the console."""
    final = result.get("final_response", {})
    sql = final.get("sql", "")
    message = final.get("message", "")
    error = final.get("error", "")
    results_table = final.get("results_table")

    print("\n" + "=" * 70)

    # Show SQL
    if sql:
        print("\n  GENERATED SQL:")
        print("  " + "-" * 50)
        for line in sql.splitlines():
            print(f"  {line}")
        print("  " + "-" * 50)

    # Show error if any
    if error:
        print(f"\n  ERROR: {error}")

    # Show message
    if message:
        print(f"\n  {message}")

    # Show results table using pandas
    if results_table:
        print("\n  RESULTS:")
        print("  " + "-" * 50)
        try:
            df = pd.DataFrame(results_table)
            with pd.option_context(
                "display.max_rows", 50,
                "display.max_columns", 20,
                "display.width", 200,
                "display.max_colwidth", 40,
            ):
                formatted = df.to_string(index=False)
                for line in formatted.splitlines():
                    print(f"  {line}")
        except Exception:
            # Fallback: print rows as dicts
            for row in results_table[:20]:
                print(f"  {row}")
            if len(results_table) > 20:
                print(f"  ... ({len(results_table)} total rows)")
        print("  " + "-" * 50)

    print("\n" + "=" * 70 + "\n")


def interactive_mode():
    """Run the agent in an interactive loop."""
    print("=" * 70)
    print("  Natural Language to SQL Data Query Agent")
    print("  Database: Sakila (MySQL via XAMPP)")
    print("  Powered by: LangGraph + Groq + FAISS + SentenceTransformers")
    print()
    print("  Type your question in plain English.")
    print("  Type 'exit', 'quit', or 'bye' to stop.")
    print("=" * 70 + "\n")

    while True:
        try:
            user_q = input("Ask your question: ").strip()
            if not user_q:
                continue
            if user_q.lower() in ("exit", "quit", "bye"):
                print("\nGoodbye!")
                break
            result = run_query(user_q)
            display_result(result)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


def single_question_mode(question: str):
    """Run a single question and exit."""
    result = run_query(question)
    display_result(result)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single question passed as command-line argument
        question = " ".join(sys.argv[1:])
        single_question_mode(question)
    else:
        # Interactive mode
        interactive_mode()
