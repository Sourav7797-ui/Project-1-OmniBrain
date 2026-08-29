from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between nodes in the OmniBrain
    LangGraph workflow.
    """

    # -------------------------
    # User input
    # -------------------------
    user_query: str
    conversation_history: list[dict[str, Any]]

    # -------------------------
    # Supervisor decisions
    # -------------------------
    intent: str
    selected_agents: list[str]
    current_agent_index: int
    next_agent: str

    # -------------------------
    # Specialist agent outputs
    # -------------------------
    search_results: list[dict[str, Any]]
    sql_results: dict[str, Any]
    vision_results: list[dict[str, Any]]

    # -------------------------
    # Images supplied by
    # the ingestion pipeline
    # -------------------------
    images: list[dict[str, Any]]

    # -------------------------
    # Combined context
    # -------------------------
    retrieved_context: list[dict[str, Any]]
    citations: list[dict[str, Any]]

    # -------------------------
    # Self-RAG
    # -------------------------
    relevance_score: float
    grounded: bool
    retry_count: int
    rewritten_query: str

    # -------------------------
    # Response
    # -------------------------
    intermediate_answer: str
    final_answer: str

    # -------------------------
    # Errors
    # -------------------------
    error: str | None