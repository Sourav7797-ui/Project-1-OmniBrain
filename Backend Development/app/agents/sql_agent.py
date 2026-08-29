from typing import Any


class SQLAgent:
    """
    Quantitative reasoning agent.

    The SQL generation model and database are injected so that
    this agent does not depend directly on a particular database.
    """

    name = "sql_agent"

    def __init__(
        self,
        database: Any = None,
        sql_generator: Any = None,
    ):
        self.database = database
        self.sql_generator = sql_generator

    async def run(self, query: str) -> dict[str, Any]:
        """
        Convert a quantitative question into a safe database operation.
        """

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

        if self.sql_generator is None:
            return {
                "success": False,
                "agent": self.name,
                "result": None,
                "error": "SQL generator is not connected.",
            }

        try:
            sql = await self.sql_generator.generate(query)

            if not self._is_read_only(sql):
                return {
                    "success": False,
                    "agent": self.name,
                    "result": None,
                    "error": "Unsafe SQL operation rejected.",
                }

            result = await self.database.execute_read_only(sql)

            return {
                "success": True,
                "agent": self.name,
                "sql": sql,
                "result": result,
            }

        except Exception as exc:
            return {
                "success": False,
                "agent": self.name,
                "result": None,
                "error": str(exc),
            }

    @staticmethod
    def _is_read_only(sql: str) -> bool:
        """
        Basic first-line SQL safety check.

        The production version should use a proper SQL parser
        and database permissions as an additional safety layer.
        """

        if not sql:
            return False

        normalized = sql.strip().lower()

        allowed = (
            normalized.startswith("select"),
            normalized.startswith("with"),
        )

        forbidden_keywords = (
            "insert ",
            "update ",
            "delete ",
            "drop ",
            "alter ",
            "create ",
            "truncate ",
            "grant ",
            "revoke ",
        )

        if not any(allowed):
            return False

        return not any(
            keyword in normalized
            for keyword in forbidden_keywords
        )