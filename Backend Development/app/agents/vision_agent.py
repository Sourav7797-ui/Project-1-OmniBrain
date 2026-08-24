from typing import Any


class VisionAgent:
    """
    Agent responsible for reasoning over visual content
    extracted from uploaded documents.
    """

    name = "vision_agent"

    def __init__(self, vision_model: Any = None):
        self.vision_model = vision_model

    async def run(
        self,
        query: str,
        images: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Answer questions about charts, diagrams, tables,
        and other visual document content.
        """

        if not query or not query.strip():
            return {
                "success": False,
                "agent": self.name,
                "results": [],
                "error": "Query cannot be empty.",
            }

        images = images or []

        if self.vision_model is None:
            return {
                "success": True,
                "agent": self.name,
                "results": [],
                "message": "Vision model is not connected yet.",
            }

        results = []

        for image in images:
            result = await self.vision_model.analyze(
                image=image,
                query=query,
            )

            results.append(result)

        return {
            "success": True,
            "agent": self.name,
            "results": results,
        }