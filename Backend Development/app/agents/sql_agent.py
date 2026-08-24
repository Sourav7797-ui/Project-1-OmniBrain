from typing import Any


class SQLAgent:
    """Agent responsible for quantitative database questions."""

    name = "sql_agent"

    def __init__(self, database: Any = None):
        self.database = database

    async def run(self, query: str) -> dict[str, Any]:
        """Process a quantitative user query."""

        if not query or not query.strip():
            return {
                "success": False,
                "agent": self.name,
                "result": None,
                "error": "Query cannot be empty.",
            }

        if self.database is None:
            return {
                "success": True,
                "agent": self.name,
                "result": None,
                "message": "Database is not connected yet.",
            }

        result = await self.database.execute_read_only(query)

        return {
            "success": True,
            "agent": self.name,
            "result": result,
        }