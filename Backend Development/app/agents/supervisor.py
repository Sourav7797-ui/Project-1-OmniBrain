from typing import Any

from .state import AgentState


# ============================================================
# Agent names
# ============================================================

SEARCH_AGENT = "search_agent"
SQL_AGENT = "sql_agent"
VISION_AGENT = "vision_agent"


# ============================================================
# Intent classification
# ============================================================

def classify_intent(query: str) -> tuple[str, list[str]]:
    """
    Classify the user's query and determine which specialist
    agent or agents should handle it.

    This is a development-time classifier.

    Later, this can be replaced with an LLM-based structured
    intent classifier.
    """

    text = query.lower().strip()

    visual_keywords = [
        "chart",
        "graph",
        "plot",
        "image",
        "diagram",
        "figure",
        "visual",
        "infographic",
    ]

    sql_keywords = [
        "revenue",
        "profit",
        "loss",
        "sales",
        "growth",
        "percentage",
        "percent",
        "amount",
        "total",
        "average",
        "maximum",
        "minimum",
        "compare",
        "financial",
        "2024",
        "2025",
        "2026",
    ]

    has_visual = any(
        keyword in text
        for keyword in visual_keywords
    )

    has_sql = any(
        keyword in text
        for keyword in sql_keywords
    )

    # --------------------------------------------------------
    # Mixed query
    # --------------------------------------------------------

    if has_visual and has_sql:
        return "mixed", [
            SQL_AGENT,
            VISION_AGENT,
        ]

    # --------------------------------------------------------
    # Visual query
    # --------------------------------------------------------

    if has_visual:
        return "visual", [
            VISION_AGENT,
        ]

    # --------------------------------------------------------
    # Quantitative query
    # --------------------------------------------------------

    if has_sql:
        return "quantitative", [
            SQL_AGENT,
        ]

    # --------------------------------------------------------
    # Default: document search
    # --------------------------------------------------------

    return "document_search", [
        SEARCH_AGENT,
    ]


# ============================================================
# Supervisor node
# ============================================================

async def supervisor_node(
    state: AgentState,
) -> dict[str, Any]:
    """
    Decide which specialist agent should execute next.
    """

    query = state.get(
        "user_query",
        "",
    ).strip()

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query:
        return {
            "error": "User query cannot be empty.",
            "selected_agents": [],
            "next_agent": "",
        }

    # --------------------------------------------------------
    # First supervisor pass:
    # classify the query
    # --------------------------------------------------------

    if not state.get("selected_agents"):

        intent, agents = classify_intent(
            query
        )

        return {
            "intent": intent,
            "selected_agents": agents,
            "current_agent_index": 0,
            "next_agent": agents[0] if agents else "",
        }

    # --------------------------------------------------------
    # Existing plan:
    # move to the next agent
    # --------------------------------------------------------

    current_index = state.get(
        "current_agent_index",
        0,
    )

    agents = state.get(
        "selected_agents",
        [],
    )

    next_index = current_index + 1

    if next_index < len(agents):

        return {
            "current_agent_index": next_index,
            "next_agent": agents[next_index],
        }

    # --------------------------------------------------------
    # All selected agents have finished
    # --------------------------------------------------------

    return {
        "current_agent_index": next_index,
        "next_agent": "",
    }


# ============================================================
# Synthesis node
# ============================================================

async def synthesis_node(
    state: AgentState,
) -> dict[str, Any]:
    """
    Combine specialist agent outputs.

    This is currently a development implementation.

    Later, this node will use an LLM to synthesize a grounded
    response and attach citations.
    """

    sections: list[str] = []

    # --------------------------------------------------------
    # Search results
    # --------------------------------------------------------

    search_results = state.get(
        "search_results",
        [],
    )

    if search_results:

        sections.append(
            f"Document retrieval returned "
            f"{len(search_results)} result(s)."
        )

    # --------------------------------------------------------
    # SQL results
    # --------------------------------------------------------

    sql_results = state.get(
        "sql_results",
        {},
    )

    if sql_results:

        if sql_results.get("result") is not None:

            sections.append(
                "SQL Agent produced a quantitative result."
            )

        elif sql_results.get("message"):

            sections.append(
                sql_results["message"]
            )

    # --------------------------------------------------------
    # Vision results
    # --------------------------------------------------------

    vision_results = state.get(
        "vision_results",
        [],
    )

    if vision_results:

        sections.append(
            f"Vision Agent analyzed "
            f"{len(vision_results)} visual item(s)."
        )

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not sections:

        sections.append(
            "No external agent data is available yet."
        )

    answer = " ".join(sections)

    return {
        "intermediate_answer": answer,
        "final_answer": answer,
    }