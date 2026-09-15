"""
OmniBrain Langfuse tracing/eval instrumentation.

Exposes:
    log_trace_event(trace_id, session_id, input_query, output_memo, citations) -> None

Imported lazily by Routers/chat.py:

    from Guardrails.langfuse_client import log_trace_event

Behavior:
- If `langfuse` is installed and LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
  are set, every chat turn is logged as a Langfuse trace with a
  "guardrail-check" span and a "citation-count" score, so the
  Guardrails/Eval teammate can review agent runs in the Langfuse UI.
- Otherwise, falls back to appending a JSON line to a local log file
  (./logs/traces.jsonl) so nothing is silently lost during local dev.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("omnibrain.guardrails.langfuse")

_LOCAL_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOCAL_LOG_FILE = _LOCAL_LOG_DIR / "traces.jsonl"

_client = None
_client_init_attempted = False


def _get_client():
    global _client, _client_init_attempted
    if _client_init_attempted:
        return _client
    _client_init_attempted = True

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        logger.info("Langfuse keys not set; tracing will be logged locally instead.")
        return None

    try:
        from langfuse import Langfuse  # type: ignore

        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info("Langfuse client initialized.")
    except Exception as exc:
        logger.warning("Langfuse unavailable (%s); falling back to local logging.", exc)
        _client = None

    return _client


def _write_local_fallback(record: Dict[str, Any]) -> None:
    try:
        _LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_LOCAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.error("Failed writing local trace fallback: %s", exc)


async def log_trace_event(
    trace_id: str,
    session_id: str,
    input_query: str,
    output_memo: str,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Records one chat turn as a Langfuse trace (or a local JSONL fallback
    line when Langfuse isn't configured). Never raises — telemetry
    failures must not break the chat response path.
    """
    citations = citations or []
    record = {
        "trace_id": trace_id,
        "session_id": session_id,
        "input_query": input_query,
        "output_memo": output_memo,
        "citation_count": len(citations),
        "citations": citations,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    client = _get_client()
    if client is None:
        _write_local_fallback(record)
        return

    try:
        trace = client.trace(
            id=trace_id,
            name="omnibrain-chat-turn",
            session_id=session_id,
            input=input_query,
            output=output_memo,
            metadata={"citation_count": len(citations)},
        )

        trace.span(
            name="guardrail-check",
            input=input_query,
            output=output_memo,
            metadata={"stage": "input+output rails passed"},
        )

        trace.generation(
            name="synthesized-memo",
            input=input_query,
            output=output_memo,
        )

        trace.score(
            name="citation-count",
            value=len(citations),
            comment=f"{len(citations)} citation(s) attached to memo",
        )

        client.flush()
    except Exception as exc:
        logger.warning("Langfuse logging failed (%s); writing local fallback.", exc)
        _write_local_fallback(record)
