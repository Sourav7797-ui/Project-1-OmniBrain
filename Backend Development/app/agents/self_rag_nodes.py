from typing import Any

from .state import AgentState


async def evaluate_retrieval(
    state: AgentState,
) -> dict[str, Any]:
    """
    Evaluate whether the current agent results contain
    enough information to continue.

    Development version uses simple result presence.
    Production version should use an LLM-based relevance
    and groundedness evaluator.
    """

    search_results = state.get(
        "search_results",
        [],
    )

    sql_results = state.get(
        "sql_results",
        {},
    )

    vision_results = state.get(
        "vision_results",
        [],
    )

    has_search = len(search_results) > 0

    has_sql = (
        bool(sql_results)
        and sql_results.get("result") is not None
    )

    has_vision = len(vision_results) > 0

    evidence_count = (
        int(has_search)
        + int(has_sql)
        + int(has_vision)
    )

    if evidence_count > 0:
        relevance_score = 1.0
        grounded = True
    else:
        relevance_score = 0.0
        grounded = False

    return {
        "relevance_score": relevance_score,
        "grounded": grounded,
    }


async def rewrite_query(
    state: AgentState,
) -> dict[str, Any]:
    """
    Rewrite a query when retrieval is insufficient.

    Development version adds clarification language.
    Production version should use an LLM query-rewriting node.
    """

    original_query = state.get(
        "user_query",
        "",
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    rewritten = (
        f"{original_query} "
        "Provide relevant document evidence, "
        "page references, and supporting context."
    )

    return {
        "rewritten_query": rewritten,
        "user_query": rewritten,
        "retry_count": retry_count + 1,
    }


def should_retry(state: AgentState) -> str:
    """
    Decide whether Self-RAG should retry retrieval.
    """

    grounded = state.get(
        "grounded",
        False,
    )

    retry_count = state.get(
        "retry_count",
        0,
    )

    if grounded:
        return "synthesize"

    if retry_count >= 2:
        return "synthesize"

    return "rewrite"