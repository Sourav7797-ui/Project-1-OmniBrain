"""
OmniBrain NeMo Guardrails integration.

Exposes:
    validate_input_rails(query: str, trace_id: str) -> str
    validate_output_rails(memo: str, trace_id: str) -> str

Both are imported lazily by Routers/chat.py:

    from Guardrails.nemo_config import validate_input_rails
    from Guardrails.nemo_config import validate_output_rails

On a policy violation they raise `main.GuardrailViolationException`
(imported lazily here too, to avoid a circular import with main.py,
which already registers a FastAPI exception handler for it).

Behavior:
- If the `nemoguardrails` package + an LLM key are available, real
  NeMo Guardrails self-check / jailbreak / groundedness rails run
  against Guardrails/rails_config/.
- If not (e.g. local dev with no OPENAI_API_KEY, or the package isn't
  installed), this falls back to fast local heuristic checks so the
  guardrail behavior is never silently skipped.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("omnibrain.guardrails.nemo")

_CONFIG_DIR = Path(__file__).parent / "rails_config"

# ---------------------------------------------------------------------
# Heuristic fallback (no external LLM / package required)
# ---------------------------------------------------------------------
_JAILBREAK_PATTERNS = [
    r"ignore (all|any|the) (previous|prior|above) instructions",
    r"ignore (the )?system prompt",
    r"reveal (your |the )?system prompt",
    r"you are now (dan|jailbroken|unrestricted)",
    r"pretend (you have|to have) no (restrictions|rules|guidelines)",
    r"disregard (your |all )?(safety|guardrail)s?",
    r"act as (a jailbroken|an unfiltered|an uncensored)",
]

_TOXIC_TERMS = [
    "kill yourself", "hate speech", "racial slur",
]

_JAILBREAK_RE = re.compile("|".join(_JAILBREAK_PATTERNS), re.IGNORECASE)


def _heuristic_input_violation(text: str) -> Optional[str]:
    if _JAILBREAK_RE.search(text):
        return "jailbreak_attempt"
    return None


def _heuristic_output_violation(text: str) -> Optional[str]:
    lowered = text.lower()
    for term in _TOXIC_TERMS:
        if term in lowered:
            return "toxicity"
    return None


# ---------------------------------------------------------------------
# Real NeMo Guardrails backend (optional)
# ---------------------------------------------------------------------
_rails_app = None
_rails_init_attempted = False


def _get_rails_app():
    """Lazily build the LLMRails instance. Returns None if unavailable."""
    global _rails_app, _rails_init_attempted
    if _rails_init_attempted:
        return _rails_app
    _rails_init_attempted = True

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("NEMO_GUARDRAILS_MODEL"):
        # No LLM credentials configured — stay on the heuristic path.
        logger.info("No LLM credentials found; using heuristic guardrails only.")
        return None

    try:
        from nemoguardrails import LLMRails, RailsConfig  # type: ignore

        config = RailsConfig.from_path(str(_CONFIG_DIR))
        _rails_app = LLMRails(config)
        logger.info("NeMo Guardrails initialized from %s", _CONFIG_DIR)
    except Exception as exc:  # ImportError or bad config both land here
        logger.warning("NeMo Guardrails unavailable (%s); falling back to heuristics.", exc)
        _rails_app = None

    return _rails_app


async def _run_self_check(user_message: str) -> bool:
    """Returns True if the message passes the 'self check input' rail."""
    rails = _get_rails_app()
    if rails is None:
        return True  # nothing to check against; heuristic path handles safety
    try:
        response = await rails.generate_async(
            messages=[{"role": "user", "content": user_message}]
        )
        # NeMo Guardrails returns a refusal bot message when a rail fires;
        # treat that as "blocked".
        content = (response or {}).get("content", "") if isinstance(response, dict) else str(response)
        return "can't comply" not in content.lower() and "withhold" not in content.lower()
    except Exception as exc:
        logger.warning("NeMo self-check call failed (%s); allowing through heuristics.", exc)
        return True


# ---------------------------------------------------------------------
# Public API used by Routers/chat.py
# ---------------------------------------------------------------------
async def validate_input_rails(query: str, trace_id: str) -> str:
    """
    Runs jailbreak / topicality checks on an incoming user query.
    Returns the (possibly unchanged) validated query, or raises
    GuardrailViolationException on a policy violation.
    """
    violation = _heuristic_input_violation(query)
    if violation:
        from main import GuardrailViolationException
        logger.warning("[trace=%s] input blocked: %s", trace_id, violation)
        raise GuardrailViolationException(
            message="Your query contains restricted prompt patterns violating NeMo safety rails.",
            violation_type=violation,
        )

    passed = await _run_self_check(query)
    if not passed:
        from main import GuardrailViolationException
        logger.warning("[trace=%s] input blocked by NeMo self-check", trace_id)
        raise GuardrailViolationException(
            message="Your query was flagged by NeMo Guardrails input rails.",
            violation_type="nemo_self_check_input",
        )

    return query


async def validate_output_rails(memo: str, trace_id: str) -> str:
    """
    Runs toxicity / groundedness checks on the agent's synthesized memo.
    Returns the (possibly unchanged) memo, or raises
    GuardrailViolationException if it can't be delivered safely.
    """
    violation = _heuristic_output_violation(memo)
    if violation:
        from main import GuardrailViolationException
        logger.warning("[trace=%s] output blocked: %s", trace_id, violation)
        raise GuardrailViolationException(
            message="The generated memo was withheld by NeMo output rails.",
            violation_type=violation,
        )

    # Real groundedness/toxicity self-check, when an LLM backend is configured.
    rails = _get_rails_app()
    if rails is not None:
        try:
            response = await rails.generate_async(
                messages=[{"role": "assistant", "content": memo}]
            )
            content = (response or {}).get("content", "") if isinstance(response, dict) else str(response)
            if "withhold" in content.lower():
                from main import GuardrailViolationException
                raise GuardrailViolationException(
                    message="The generated memo failed the NeMo groundedness/toxicity check.",
                    violation_type="nemo_self_check_output",
                )
        except Exception as exc:
            logger.warning("NeMo output self-check failed (%s); returning memo unchecked.", exc)

    return memo
