"""
graph.py — LangGraph StateGraph Workflow

Assembles the complete Natural Language to SQL agent workflow using a single
LangGraph StateGraph.  All nodes are imported from nodes.py and wired together
with conditional edges for SQL validation retry logic.

Workflow:
  understand_question → retrieve_schema → expand_relationships → generate_sql
      → validate_sql ──[valid]──→ execute_sql → format_response
                       ──[retry]──→ generate_sql  (loop, max 3)
                       ──[fail]───→ format_response
"""

from langgraph.graph import StateGraph, END

from models import AgentState
from nodes import (
    understand_question,
    retrieve_schema,
    expand_relationships,
    generate_sql,
    validate_sql,
    execute_sql,
    format_response,
    should_retry_sql,
)


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph agent workflow."""

    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("understand_question", understand_question)
    workflow.add_node("retrieve_schema", retrieve_schema)
    workflow.add_node("expand_relationships", expand_relationships)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("validate_sql", validate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("format_response", format_response)

    # Set entry point
    workflow.set_entry_point("understand_question")

    # Linear edges
    workflow.add_edge("understand_question", "retrieve_schema")
    workflow.add_edge("retrieve_schema", "expand_relationships")
    workflow.add_edge("expand_relationships", "generate_sql")
    workflow.add_edge("generate_sql", "validate_sql")

    # Conditional edge after validation
    workflow.add_conditional_edges(
        "validate_sql",
        should_retry_sql,
        {
            "execute": "execute_sql",
            "retry": "generate_sql",
            "fail": "format_response",
        }
    )

    # Terminal edges
    workflow.add_edge("execute_sql", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()


# Compiled graph — importable singleton
agent_graph = build_graph()
