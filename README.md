# OmniBrain

**Agentic Multi-Modal RAG Orchestrator** — an internship project by Axlero Innovating Solutions.

OmniBrain lets a financial analyst upload a long PDF (e.g. a 500-page annual report), ask questions about it in natural language, and get back a **synthesized, cited investment memo** — grounded in the document's text, tables, and charts, with a safety layer checking every input and output.

---

## What it does

1. An analyst uploads a PDF through the web UI.
2. The backend parses it into text, tables, and images/charts, chunks it with page-level metadata, and embeds it into a vector store.
3. When the analyst asks a question, a **supervisor agent** routes the query to one or more specialist agents:
   - a semantic **search agent** over the document's text/embeddings,
   - a text-to-**SQL agent** over historical stock data,
   - a **vision agent** that reasons over extracted charts and tables.
4. The agents' findings are synthesized into a memo.
5. Both the incoming query and the outgoing memo pass through a **guardrails layer** (jailbreak/toxicity/groundedness checks) before the analyst sees anything.
6. Every chat turn is traced for observability/eval.

---

## Architecture

```
Analyst (browser)
      │
      ▼
Frontend (Streamlit)  ──REST/WebSocket──►  Backend (FastAPI)
                                                  │
                        ┌─────────────────────────┼─────────────────────────┐
                        ▼                          ▼                         ▼
                   Ingestion                   Agents                  Guardrails
              (parse → chunk → embed)     (LangGraph supervisor +      (input/output rails,
                        │                  search/sql/vision agents)    Langfuse tracing)
                        ▼                          │
                  Data layer                       │
             (Qdrant/FAISS + SQL DB) ◄──────────────┘
                        │
                        ▼
              Cited Investment Memo → Analyst
```

The Frontend never talks to the database, vector store, or agents directly — every request goes through the Backend's REST/WebSocket API. This separation is what let the 5-person team build each layer independently.

---

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI |
| Agent orchestration | LangGraph |
| Vector store | Qdrant (FAISS-compatible) |
| Guardrails | NeMo Guardrails |
| Observability / eval | Langfuse |
| Relational DB | PostgreSQL (async, via SQLAlchemy) / SQLite for local dev |
| PDF parsing | PyMuPDF (`fitz`) + a vision-language model (`minicpm-v` via Ollama) for tables/charts |
| Auth | JWT (`python-jose`) + `passlib`/bcrypt |

---

## Project structure

```
OmniBrain/
├── Frontend Development/
│   ├── app.py                  # Streamlit entry point
│   ├── pages/
│   │   ├── chat.py             # Query input + streamed response + citations
│   │   ├── upload.py           # PDF upload widget + ingestion progress
│   │   ├── admin.py            # Admin controls
│   │   └── profile.py          # User profile / session info
│   └── Utils/
│       └── api.py              # Centralized HTTP/WebSocket client
│
├── Backend Development/
│   ├── main.py                 # FastAPI app instance, router registration   ← entrypoint
│   ├── auth.py                 # Auth/authorization logic
│   ├── Routers/
│   │   ├── auth.py             # POST /login, /register
│   │   ├── upload.py           # POST /upload, GET /status/{job_id}
│   │   ├── chat.py             # POST /chat, WS /chat/stream, GET /history
│   │   └── admin.py            # /metrics, /docs, /users
│   ├── Ingestion/
│   │   ├── pdf_extractor.py    # Splits PDF into text/tables/images
│   │   ├── extractor.py        # Table/image summarization via VLM
│   │   ├── chunker.py          # Chunks text with page/section metadata
│   │   ├── embedder.py         # Embeds chunks, writes to vector store
│   │   └── pipeline.py         # Wires the above into one ingestion run
│   ├── app/agents/
│   │   ├── supervisor.py       # LangGraph supervisor — shared state + routing
│   │   ├── graph.py            # LangGraph graph definition
│   │   ├── search_agent.py     # Semantic retrieval over the vector DB
│   │   ├── sql_agent.py        # Text-to-SQL over historical stock data
│   │   └── vision_agent.py     # VLM reasoning over charts/tables
│   ├── Guardrails/
│   │   ├── nemo_config.py      # Input/output safety rails (jailbreak, toxicity, groundedness)
│   │   ├── langfuse_client.py  # Trace/eval instrumentation for every agent call
│   │   └── rails_config/       # NeMo Guardrails Colang config
│   └── Database/
│       ├── database.py         # Async SQL session management
│       ├── models.py           # SQLAlchemy ORM models
│       ├── schemas.py          # Pydantic request/response schemas
│       ├── vector_store.py     # Qdrant client wrapper
│       ├── crud.py             # Reusable SQL CRUD helpers
│       └── docker-compose.yml  # Local Qdrant container
│
├── setup_and_run.sh            # One-shot install + start script
└── test_guardrails_standalone.py  # Zero-dependency test for the Guardrails module
```

---

## API contract

| Endpoint | Method | Purpose |
|---|---|---|
| `/login`, `/register` | POST | Auth |
| `/upload` | POST | Accepts a PDF, returns a `job_id`, triggers async ingestion |
| `/status/{job_id}` | GET | Poll ingestion progress |
| `/chat` | POST | Send a query, returns a synthesized memo + citations |
| `/chat/stream` | WS | Streamed token-by-token agent response |
| `/history` | GET | Past queries/memos for a session |
| `/metrics`, `/docs`, `/users` | GET/POST/DELETE | Admin: system metrics, document management, user management |

Full interactive docs are available at `/docs` (Swagger UI) once the backend is running.

---

## Getting started

### Prerequisites
- Python 3.10+
- (Optional) Docker, if you want a local Qdrant instance instead of a hosted one
- (Optional) [Ollama](https://ollama.com) running locally with the `minicpm-v` model pulled, for table/chart summarization during ingestion

### 1. Configure environment variables
```bash
cp "Backend Development/Database/.env.example" "Backend Development/.env"
```
Fill in, at minimum:
- `DATABASE_URL` — Postgres URL, or `sqlite+aiosqlite:///./omnibrain.db` for local dev
- `APP_SECRET_KEY` — any random string, used for JWT signing
- `QDRANT_HOST` / `QDRANT_PORT` — only needed if you're running the vector store (see step 2)

Guardrails are optional too — `OPENAI_API_KEY` enables real NeMo self-check rails, and `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` enable Langfuse tracing. Without either, the app runs on local heuristic checks and a JSONL trace log instead.

### 2. (Optional) Start Qdrant locally
```bash
cd "Backend Development/Database"
docker compose up -d
```

### 3. Install everything and start the backend
From the repo root:
```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```
This creates a virtualenv, installs every `requirements*.txt` in the repo plus a few packages the code imports but that aren't captured in any requirements file, and starts FastAPI at `http://localhost:8000` (Swagger docs at `/docs`).

Add `--with-frontend` to also launch the Streamlit UI:
```bash
./setup_and_run.sh --with-frontend
```

### 4. Sanity-check the Guardrails layer on its own
No install required:
```bash
python3 test_guardrails_standalone.py
```


