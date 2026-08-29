from typing import Any


class SearchAgent:
    """
    Semantic document retrieval agent.

    The actual Qdrant/vector-store implementation is injected
    through vector_store.
    """

    name = "search_agent"

    def __init__(self, vector_store: Any = None):
        self.vector_store = vector_store

    async def run(self, query: str) -> dict[str, Any]:
        """
        Retrieve relevant document chunks.
        """

        if not query or not query.strip():
            return {
                "success": False,
                "agent": self.name,
                "results": [],
                "error": "Query cannot be empty.",
            }

        # Development mode: no vector store connected yet.
        if self.vector_store is None:
            return {
                "success": True,
                "agent": self.name,
                "results": [],
                "message": "Vector store is not connected yet.",
            }

        try:
            results = await self.vector_store.search(query)

            return {
                "success": True,
                "agent": self.name,
                "results": results or [],
            }

        except Exception as exc:
            return {
                "success": False,
                "agent": self.name,
                "results": [],
                "error": str(exc),
            }