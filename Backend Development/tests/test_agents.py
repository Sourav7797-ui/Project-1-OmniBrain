import pytest

from app.agents.search_agent import SearchAgent
from app.agents.sql_agent import SQLAgent
from app.agents.vision_agent import VisionAgent


@pytest.mark.asyncio
async def test_search_agent_empty_query():
    agent = SearchAgent()

    result = await agent.run("")

    assert result["success"] is False
    assert result["error"] == "Query cannot be empty."


@pytest.mark.asyncio
async def test_search_agent_without_vector_store():
    agent = SearchAgent()

    result = await agent.run("What is the revenue?")

    assert result["success"] is True
    assert result["agent"] == "search_agent"


@pytest.mark.asyncio
async def test_sql_agent_empty_query():
    agent = SQLAgent()

    result = await agent.run("")

    assert result["success"] is False
    assert result["error"] == "Query cannot be empty."


@pytest.mark.asyncio
async def test_sql_agent_without_database():
    agent = SQLAgent()

    result = await agent.run("What was revenue in 2025?")

    assert result["success"] is True
    assert result["agent"] == "sql_agent"


@pytest.mark.asyncio
async def test_vision_agent_empty_query():
    agent = VisionAgent()

    result = await agent.run("")

    assert result["success"] is False
    assert result["error"] == "Query cannot be empty."


@pytest.mark.asyncio
async def test_vision_agent_without_model():
    agent = VisionAgent()

    result = await agent.run(
        "What does this chart show?",
        [],
    )

    assert result["success"] is True
    assert result["agent"] == "vision_agent"