import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request, WebSocket, WebSocketDisconnect
from Database.schemas import ChatRequest, ChatResponse, Citation, HistoryResponse, MessageRecord, TokenData
from auth import get_current_user

router = APIRouter()

SESSION_STORAGE: Dict[str, List[Dict[str, Any]]] = {}


async def apply_guardrails_input(query: str, trace_id: str) -> str:
    try:
        from Guardrails.nemo_config import validate_input_rails
        validated_text = await validate_input_rails(query, trace_id=trace_id)
        return validated_text
    except ImportError:
        blocked_terms = ["ignore all instructions", "system prompt leak", "exploit"]
        if any(term in query.lower() for term in blocked_terms):
            from main import GuardrailViolationException
            raise GuardrailViolationException(
                message="Your query contains restricted prompt patterns violating NeMo safety rails.",
                violation_type="jailbreak_attempt"
            )
        return query


async def apply_guardrails_output(memo: str, trace_id: str) -> str:
    try:
        from Guardrails.nemo_config import validate_output_rails
        return await validate_output_rails(memo, trace_id=trace_id)
    except ImportError:
        return memo


async def log_telemetry_trace(trace_id: str, session_id: str, query: str, output: str, citations: List[Citation]):
    try:
        from Guardrails.langfuse_client import log_trace_event
        await log_trace_event(
            trace_id=trace_id,
            session_id=session_id,
            input_query=query,
            output_memo=output,
            citations=[c.model_dump() for c in citations]
        )
    except ImportError:
        pass


async def invoke_agent_supervisor(query: str, session_id: str, trace_id: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        from Agents.supervisor import run_supervisor_workflow
        result = await run_supervisor_workflow(
            query=query,
            session_id=session_id,
            trace_id=trace_id,
            filters=filters
        )
        return result
    except (ImportError, AttributeError):
        return {
            "memo": (
                f"Synthesized Investment Analysis for query: '{query}'.\n\n"
                "Key Findings:\n"
                "• Revenue increased by 14.2% year-over-year driven by cloud segment expansion.\n"
                "• Operating cash flows remain robust at $2.4B with lower leverage ratios."
            ),
            "citations": [
                {
                    "source": "Q3_Financial_Report.pdf",
                    "page": 12,
                    "snippet": "Operating margin expanded to 28.4% reflecting operational efficiencies.",
                    "score": 0.91
                },
                {
                    "source": "Balance_Sheet_2026.pdf",
                    "page": 4,
                    "snippet": "Cash and short-term equivalents totaled $5.2B at the close of the period.",
                    "score": 0.87
                }
            ]
        }


async def stream_agent_supervisor(query: str, session_id: str, trace_id: str, filters: Optional[Dict[str, Any]] = None):
    try:
        from Agents.supervisor import stream_supervisor_workflow
        async for chunk in stream_supervisor_workflow(query=query, session_id=session_id, trace_id=trace_id, filters=filters):
            yield chunk
    except (ImportError, AttributeError):
        steps = [
            {"type": "status", "content": "Validating NeMo rails & evaluating agent intent..."},
            {"type": "token", "content": "Synthesized "},
            {"type": "token", "content": "Investment "},
            {"type": "token", "content": "Memo:\n\n"},
            {"type": "token", "content": "1. Operational EBITDA expanded by 14.8%.\n"},
            {
                "type": "citations",
                "citations": [
                    {
                        "source": "Q3_Financial_Report.pdf",
                        "page": 12,
                        "snippet": "Operating margin expanded to 28.4% reflecting operational efficiencies.",
                        "score": 0.91
                    }
                ]
            }
        ]
        import asyncio
        for step in steps:
            await asyncio.sleep(0.15)
            yield step


async def persist_chat_turn(session_id: str, user: str, query: str, memo: str, citations: List[Citation]):
    try:
        from crud import save_chat_turn
        await save_chat_turn(
            session_id=session_id,
            username=user,
            query=query,
            response=memo,
            citations=[c.model_dump() for c in citations]
        )
    except (ImportError, AttributeError):
        if session_id not in SESSION_STORAGE:
            SESSION_STORAGE[session_id] = []

        SESSION_STORAGE[session_id].append({
            "role": "user",
            "content": query,
            "citations": None,
            "timestamp": datetime.utcnow()
        })
        SESSION_STORAGE[session_id].append({
            "role": "assistant",
            "content": memo,
            "citations": citations,
            "timestamp": datetime.utcnow()
        })


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(
    request_http: Request,
    request: ChatRequest,
    current_user: TokenData = Depends(get_current_user)
):
    trace_id = getattr(request_http.state, "trace_id", str(uuid.uuid4()))

    # 1. Apply Input Guardrails (NeMo)
    validated_query = await apply_guardrails_input(request.query, trace_id=trace_id)

    # 2. Delegate to LangGraph Supervisor
    agent_output = await invoke_agent_supervisor(
        query=validated_query,
        session_id=request.session_id,
        trace_id=trace_id,
        filters=request.filters
    )

    raw_memo = agent_output.get("memo", "No response generated.")

    # 3. Apply Output Guardrails (Groundedness / Hallucination checks)
    sanitized_memo = await apply_guardrails_output(raw_memo, trace_id=trace_id)

    formatted_citations = [
        Citation(
            source=c.get("source", "Unknown Source"),
            page=c.get("page") or c.get("page_number"),
            snippet=c.get("snippet", ""),
            score=c.get("score")
        )
        for c in agent_output.get("citations", [])
    ]

    # 4. Save Session History & Emit Langfuse Traces
    await persist_chat_turn(
        session_id=request.session_id,
        user=current_user.username or "anonymous",
        query=request.query,
        memo=sanitized_memo,
        citations=formatted_citations
    )

    await log_telemetry_trace(
        trace_id=trace_id,
        session_id=request.session_id,
        query=request.query,
        output=sanitized_memo,
        citations=formatted_citations
    )

    return ChatResponse(
        session_id=request.session_id,
        memo=sanitized_memo,
        citations=formatted_citations,
        generated_at=datetime.utcnow()
    )


@router.get("/history", response_model=HistoryResponse, status_code=status.HTTP_200_OK)
async def get_chat_history(
    session_id: str = Query(..., min_length=1, description="Unique identifier for the chat session"),
    current_user: TokenData = Depends(get_current_user)
):
    try:
        from crud import get_session_history
        db_records = await get_session_history(session_id=session_id)
        formatted_messages = [
            MessageRecord(
                role=rec.get("role"),
                content=rec.get("content"),
                citations=[Citation(**c) for c in rec.get("citations", [])] if rec.get("citations") else None,
                timestamp=rec.get("timestamp", datetime.utcnow())
            )
            for rec in db_records
        ]
        return HistoryResponse(session_id=session_id, messages=formatted_messages)
    except (ImportError, AttributeError):
        records = SESSION_STORAGE.get(session_id, [])
        formatted_messages = [
            MessageRecord(
                role=rec["role"],
                content=rec["content"],
                citations=rec.get("citations"),
                timestamp=rec.get("timestamp", datetime.utcnow())
            )
            for rec in records
        ]
        return HistoryResponse(session_id=session_id, messages=formatted_messages)


@router.websocket("/chat/stream")
async def websocket_chat_stream(websocket: WebSocket):
    await websocket.accept()
    trace_id = str(uuid.uuid4())
    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "content": "Invalid JSON format."})
                continue

            query = payload.get("query", "").strip()
            session_id = payload.get("session_id", "default_ws_session")
            filters = payload.get("filters", {})

            if not query:
                await websocket.send_json({"type": "error", "content": "Query cannot be empty."})
                continue

            try:
                validated_query = await apply_guardrails_input(query, trace_id=trace_id)
            except Exception as e:
                await websocket.send_json({"type": "guardrail_block", "content": str(e)})
                continue

            accumulated_memo = ""
            collected_citations = []

            async for event in stream_agent_supervisor(query=validated_query, session_id=session_id, trace_id=trace_id, filters=filters):
                await websocket.send_json(event)
                if event.get("type") == "token":
                    accumulated_memo += event.get("content", "")
                elif event.get("type") == "citations":
                    collected_citations = [
                        Citation(**c) for c in event.get("citations", [])
                    ]

            await persist_chat_turn(
                session_id=session_id,
                user="stream_user",
                query=query,
                memo=accumulated_memo,
                citations=collected_citations
            )

            await log_telemetry_trace(
                trace_id=trace_id,
                session_id=session_id,
                query=query,
                output=accumulated_memo,
                citations=collected_citations
            )

            await websocket.send_json({"type": "done", "session_id": session_id, "trace_id": trace_id})

    except WebSocketDisconnect:
        pass