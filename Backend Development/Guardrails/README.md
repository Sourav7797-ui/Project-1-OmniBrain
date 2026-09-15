# Guardrails/ — drop-in instructions

This folder was missing from the repo even though `Routers/chat.py` and
`main.py` already reference it. Drop it in as-is:

```
Backend Development/
└── Guardrails/          <- copy this whole folder here
    ├── __init__.py
    ├── nemo_config.py
    ├── langfuse_client.py
    ├── requirements.txt
    └── rails_config/
        ├── config.yml
        ├── prompts.yml
        └── rails/
            ├── input_rails.co
            └── output_rails.co
```

No changes are needed in `Routers/chat.py` or `main.py` — the imports
already match:

```python
from Guardrails.nemo_config import validate_input_rails, validate_output_rails
from Guardrails.langfuse_client import log_trace_event
```

## Env vars (add to your `.env`)

```
# Optional — omit to run on local heuristic guardrails only
OPENAI_API_KEY=

# Optional — omit to log traces to ./logs/traces.jsonl instead of Langfuse
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Install

```
pip install -r Guardrails/requirements.txt
```

## Behavior without any keys set

Both modules degrade gracefully instead of throwing:
- `nemo_config.py` uses regex-based jailbreak/toxicity checks.
- `langfuse_client.py` appends JSON lines to `logs/traces.jsonl`.

This means the app works identically to how it does today, but now
actually runs real checks/logging instead of relying on chat.py's
`except ImportError` stub path.
