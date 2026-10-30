"""
models.py

Pydantic models for structured data and the LangGraph shared state (TypedDict).
Defines the question understanding schema, SQL validation results, and the
complete agent state passed between LangGraph nodes.
"""

from typing import TypedDict, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# Pydantic Models — Structured LLM Outputs
# ==============================================================================

class QuestionUnderstanding(BaseModel):
    """Structured representation of a parsed user question produced by the LLM."""
    intent: str = Field(description="The high-level intent of the question (e.g., 'retrieve data', 'aggregate', 'count', 'compare')")
    concepts: list[str] = Field(description="Domain concepts relevant to the question (e.g., 'customer', 'film rental', 'payment amount')")
    requested_output: str = Field(description="What the user wants returned (e.g., 'list of names', 'total count', 'average amount')")
    filters: list[str] = Field(default_factory=list, description="Filter conditions mentioned (e.g., 'active customers', 'released in 2006')")
    grouping: list[str] = Field(default_factory=list, description="Grouping dimensions (e.g., 'by category', 'per store')")
    sorting: str = Field(default="", description="Sorting requirements (e.g., 'highest first', 'alphabetical')")
    aggregations: list[str] = Field(default_factory=list, description="Aggregation operations needed (e.g., 'COUNT', 'SUM', 'AVG')")
    is_relevant: bool = Field(default=True, description="Whether the question is relevant to the database")


class SQLValidationResult(BaseModel):
    """Result of SQL validation checks."""
    is_valid: bool = Field(description="Whether the SQL passed all validation checks")
    error_message: str = Field(default="", description="Description of the validation failure, if any")
    error_type: str = Field(default="", description="Category of error: 'unsafe_operation', 'syntax', 'multiple_statements', 'empty'")


# ==============================================================================
# LangGraph Shared State — TypedDict
# ==============================================================================

class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph workflow nodes."""

    # Input
    user_question: str

    # Understand Question node output
    question_understanding: Optional[dict]   # serialized QuestionUnderstanding

    # Retrieve Schema node output
    anchor_tables: list[dict]                # [{table_name, similarity_score}]
    relevant_schemas: list[dict]             # full table metadata dicts from the catalog

    # Expand Relationships node output
    expanded_tables: list[str]               # all relevant table names (anchors + intermediates)
    join_relationships: list[dict]           # FK join mappings for the query
    schema_context: str                      # formatted schema text for the LLM

    # Generate SQL node output
    generated_sql: str

    # Validate SQL node output
    validation_result: Optional[dict]        # serialized SQLValidationResult
    sql_correction_count: int                # number of correction attempts so far
    validation_error_history: list[str]      # previous validation errors for context

    # Execute SQL node output
    query_results: Optional[dict]            # {columns, rows, row_count, truncated}
    execution_error: str

    # Format Response node output
    final_response: dict                     # {sql, results_table, message, error}
