# Architecture (CodexVid AI)

## Overview

```
┌─────────────┐     HTTP      ┌──────────────────────────────────────────┐
│  Browser /  │ ────────────► │  FastAPI (main.py)                        │
│  API client │               │  • /health, /ready                        │
└─────────────┘               │  • /api/codexvid/*                        │
                              │  • static: learn.html, / and /learn       │
                              └───────────────┬──────────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
            ┌───────────────┐        ┌──────────────┐        ┌─────────────────┐
            │ codexvid/     │        │ services/    │        │ core/llm.py     │
            │ chunking,     │        │ video,       │        │ OpenAI-compat   │
            │ timestamp_    │        │ transcription, │        │ chat + embed    │
            │ retrieval_    │        │ aws_transcribe │        │                 │
            │ utils, chat,  │        │              │        │                 │
            │ vector_store  │        │              │        │                 │
            └───────┬───────┘        └──────┬───────┘        └────────┬────────┘
                    │                       │                        │
                    ▼                       ▼                        ▼
            data/codexvid_sessions/  uploads/                  Ollama / remote
            FAISS per session_id      temp media                 LLM HTTP API
```

## HTTP surface

| Route | Role |
|-------|------|
| `GET /`, `GET /learn` | Learn UI (`static/learn.html`) |
| `GET /health` | Liveness |
| `GET /ready` | LLM readiness |
| `POST /api/codexvid/upload` | Video upload + pipeline |
| CodexVid API (see `/docs`) | Chat, session video, exists |

No workspace router, no ChromaDB, no SQLite job queue in this build.

## Core packages

- **`app/api/codexvid.py`** — Request/response models and HTTP handlers for upload and chat.
- **`app/codexvid/`** — Transcription orchestration (`transcription.py`), **word- and sentence-level** timing (`timestamp_utils.py`), **semantic** chunking (`chunking.py`, 30–60s windows, sentence boundaries), **FAISS** metadata with `start_time`/`end_time` (`vector_store.py`), **sentence-level** refinement after chunk retrieval (`retrieval_utils.py`: batched embeddings + cosine match within chunk windows), **two-stage** grounded chat (`chat.py`: extract → JSON explanation + grounding score; uses `transcript.json` + `session_id`), teaching pack (`teaching.py`: per-chunk topic LLM, aggregate/merge/coverage, optional sentence snap; takeaways/quiz from topic summaries), session I/O (`session.py`). Persistence under `VCAI_CODEXVID_SESSIONS_DIR` / default `data/codexvid_sessions`.
- **`app/services/transcription.py`** — Whisper path and shared helpers; branches to AWS when configured.
- **`app/services/aws_transcribe.py`** — S3 upload, start job, poll, parse (when `VCAI_STT_PROVIDER=aws`).
- **`app/services/video.py`** — FFmpeg-style extraction and media utilities used by the pipeline.
- **`app/core/llm.py`** — HTTP client for chat and embeddings against a configurable OpenAI-compatible base URL.

## Configuration

Single source of truth: **`app/config.py`**, overridden by environment variables prefixed with `VCAI_` (see `.env.example`).

## Static assets

- **`static/learn.html`**, **`learn.js`**, **`learn.css`** — primary learn UI.
- **`static/index.html`** — landing as needed.

## Tests

`tests/` targets the CodexVid API, chunking, retrieval helpers, timestamp utils, AWS Transcribe parsing (mocked), and related behavior—see [TESTING.md](./TESTING.md).
