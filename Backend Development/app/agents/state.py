from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between nodes in the OmniBrain
    LangGraph agent workflow.
    """

    # Original user input
    user_query: str

    # Supervisor decisions
    intent: str
    selected_agents: list[str]

    # Specialist agent outputs
    search_results: list[dict[str, Any]]
    sql_results: dict[str, Any]
    vision_results: list[dict[str, Any]]

    # Retrieved information
    retrieved_context: list[dict[str, Any]]

    # Sources used for the final response
    citations: list[dict[str, Any]]

    # Intermediate and final responses
    intermediate_answer: str
    final_answer: str

    # Self-RAG information
    relevance_score: float
    retry_count: int

    # Error information
    error: str | None