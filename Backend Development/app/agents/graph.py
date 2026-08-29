from typing import Any

from langgraph.graph import END, START, StateGraph

from .search_agent import SearchAgent
from .sql_agent import SQLAgent
from .state import AgentState
from .supervisor import (
    SEARCH_AGENT,
    SQL_AGENT,
    VISION_AGENT,
    supervisor_node,
    synthesis_node,
)
from .vision_agent import VisionAgent
from .self_rag_nodes import (
    evaluate_retrieval,
    rewrite_query,
    should_retry,
)


class OmniBrainGraph:
    """
    LangGraph orchestration layer for OmniBrain.

    External dependencies are injected so the graph can run
    during development without Qdrant, SQL or a vision model.
    """

    def __init__(
        self,
        search_agent: SearchAgent | None = None,
        sql_agent: SQLAgent | None = None,
        vision_agent: VisionAgent | None = None,
    ):
        self.search_agent = (
            search_agent
            if search_agent is not None
            else SearchAgent()
        )

        self.sql_agent = (
            sql_agent
            if sql_agent is not None
            else SQLAgent()
        )

        self.vision_agent = (
            vision_agent
            if vision_agent is not None
            else VisionAgent()
        )

        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Build and compile the OmniBrain LangGraph workflow.
        """

        workflow = StateGraph(AgentState)

        # =====================================================
        # NODES
        # =====================================================

        workflow.add_node(
            "supervisor",
            supervisor_node,
        )

        workflow.add_node(
            "search_agent",
            self.search_node,
        )

        workflow.add_node(
            "sql_agent",
            self.sql_node,
        )

        workflow.add_node(
            "vision_agent",
            self.vision_node,
        )

        workflow.add_node(
            "evaluate_retrieval",
            evaluate_retrieval,
        )

        workflow.add_node(
            "rewrite_query",
            rewrite_query,
        )

        workflow.add_node(
            "synthesis",
            synthesis_node,
        )

        # =====================================================
        # START
        # =====================================================

        workflow.add_edge(
            START,
            "supervisor",
        )

        # =====================================================
        # SUPERVISOR ROUTING
        # =====================================================

        workflow.add_conditional_edges(
            "supervisor",
            self.route_from_supervisor,
            {
                SEARCH_AGENT: SEARCH_AGENT,
                SQL_AGENT: SQL_AGENT,
                VISION_AGENT: VISION_AGENT,
                "evaluate": "evaluate_retrieval",
                "end": END,
            },
        )

        # =====================================================
        # SPECIALIST AGENTS → SUPERVISOR
        # =====================================================

        workflow.add_edge(
            SEARCH_AGENT,
            "supervisor",
        )

        workflow.add_edge(
            SQL_AGENT,
            "supervisor",
        )

        workflow.add_edge(
            VISION_AGENT,
            "supervisor",
        )

        # =====================================================
        # SELF-RAG
        # =====================================================

        workflow.add_conditional_edges(
            "evaluate_retrieval",
            should_retry,
            {
                "rewrite": "rewrite_query",
                "synthesize": "synthesis",
            },
        )

        # Rewritten query goes back to supervisor
        workflow.add_edge(
            "rewrite_query",
            "supervisor",
        )

        # =====================================================
        # FINAL RESPONSE
        # =====================================================

        workflow.add_edge(
            "synthesis",
            END,
        )

        return workflow.compile()

    # =========================================================
    # SUPERVISOR ROUTER
    # =========================================================

    def route_from_supervisor(
        self,
        state: AgentState,
    ) -> str:
        """
        Decide which node should execute next.
        """

        if state.get("error"):
            return "end"

        next_agent = state.get(
            "next_agent",
            "",
        )

        if next_agent:
            return next_agent

        return "evaluate"

    # =========================================================
    # SEARCH NODE
    # =========================================================

    async def search_node(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Execute the Search Agent.
        """

        query = state.get(
            "rewritten_query",
            state.get("user_query", ""),
        )

        result = await self.search_agent.run(
            query
        )

        results = result.get(
            "results",
            [],
        )

        return {
            "search_results": results,
            "retrieved_context": results,
            "error": result.get("error"),
        }

    # =========================================================
    # SQL NODE
    # =========================================================

    async def sql_node(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Execute the SQL Agent.
        """

        query = state.get(
            "rewritten_query",
            state.get("user_query", ""),
        )

        result = await self.sql_agent.run(
            query
        )

        return {
            "sql_results": result,
            "error": result.get("error"),
        }

    # =========================================================
    # VISION NODE
    # =========================================================

    async def vision_node(
        self,
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Execute the Vision Agent using images supplied
        by the ingestion pipeline.
        """

        query = state.get(
            "rewritten_query",
            state.get("user_query", ""),
        )

        images = state.get(
            "images",
            [],
        )

        result = await self.vision_agent.run(
            query=query,
            images=images,
        )

        return {
            "vision_results": result.get(
                "results",
                [],
            ),
            "error": result.get("error"),
        }

    # =========================================================
    # PUBLIC GRAPH INTERFACE
    # =========================================================

    async def ainvoke(
        self,
        user_query: str,
        images: list[dict[str, Any]] | None = None,
    ) -> AgentState:
        """
        Execute the complete OmniBrain agent workflow.

        Args:
            user_query:
                The user's natural-language question.

            images:
                Optional images/charts supplied by the
                document ingestion pipeline.

        Returns:
            Final LangGraph AgentState.
        """

        initial_state: AgentState = {
            "user_query": user_query,
            "conversation_history": [],

            "selected_agents": [],
            "current_agent_index": 0,
            "next_agent": "",

            "search_results": [],
            "sql_results": {},
            "vision_results": [],

            "images": images or [],

            "retrieved_context": [],
            "citations": [],

            "relevance_score": 0.0,
            "grounded": False,
            "retry_count": 0,

            "rewritten_query": "",

            "intermediate_answer": "",
            "final_answer": "",

            "error": None,
        }

        return await self.graph.ainvoke(
            initial_state
        )


# =============================================================
# CONVENIENCE FUNCTION
# =============================================================

def create_graph(
    search_agent: SearchAgent | None = None,
    sql_agent: SQLAgent | None = None,
    vision_agent: VisionAgent | None = None,
):
    """
    Create and compile an OmniBrain LangGraph.
    """

    return OmniBrainGraph(
        search_agent=search_agent,
        sql_agent=sql_agent,
        vision_agent=vision_agent,
    ).graph