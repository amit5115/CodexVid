# Project context (CodexVid AI)

Use this file to orient contributors and AI assistants to the codebase.

## Product

- **Name:** CodexVid AI (package `app`).
- **Purpose:** Upload educational video → transcribe (word-level) → sentence timeline → **semantic** chunk (≈30–60s) → embed (FAISS) → **two-stage** LLM chat (extract → explain) with chunk RAG, **sentence-level** seek times (`retrieval_utils` + `transcript.json`), teaching pack optional **sentence** digest + **snap**, and grounding checks.

## What was removed vs older docs

Older revisions described workspace APIs, ChromaDB, SQLite jobs, scoring, and a large static SPA. **This repo revision does not include those.** If documentation elsewhere mentions them, treat it as obsolete.

## Entry points

| Entry | Purpose |
|-------|---------|
| `app/main.py` | FastAPI app: mount health + codexvid routers, static files, `/` and `/learn`. |
| `python -m app.cli serve` | Production-style server (host/port/reload). |
| `make dev` | Dev server (see Makefile). |

## Configuration

- **`app/config.py`** — Pydantic-style constants; env vars `VCAI_*`.
- **`.env.example`** — Template; copy to `.env` locally.

## Important directories

| Path | Contents |
|------|----------|
| `app/api/health.py` | `/health`, `/ready`. |
| `app/api/codexvid.py` | CodexVid HTTP API. |
| `app/codexvid/` | `transcription`, `timestamp_utils` (words/sentences), `chunking` (semantic), `retrieval_utils` (sentence pick / embeddings), `vector_store`, `chat` (multi-stage + session transcripts), `teaching` (sentence digest + snap), `session`. |
| `app/services/` | `transcription`, `video`, `aws_transcribe`. |
| `app/core/llm.py` | LLM + embedding HTTP client. |
| `app/static/` | `learn.*`, `index.html`. |
| `data/` (runtime) | Default under `VCAI_DATA_DIR`; sessions under `codexvid_sessions/` unless overridden. |

## Dependencies (conceptual)

- FastAPI, Uvicorn, Pydantic.
- faster-whisper (local STT); optional boto3 for AWS Transcribe.
- numpy, faiss-cpu (or faiss per platform), httpx.
- No SQLAlchemy/Chroma in the slim stack—check `pyproject.toml` for the exact list.

## Testing

- **`tests/`** — pytest; **28** tests (see [TESTING.md](./TESTING.md)).

## Docs index

- [README.md](./README.md) — Quick start.
- [ARCHITECTURE.md](./ARCHITECTURE.md) — Diagrams and components.
- [APP_DATAFLOW.md](./APP_DATAFLOW.md) — Pipeline steps.
- [TESTING.md](./TESTING.md) — Commands and scope.
- [UI_CLICK_GUIDE.md](./UI_CLICK_GUIDE.md) — Learn UI.
