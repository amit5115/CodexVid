# Project Context (CodexVid AI)

Use this file to orient contributors and AI assistants to the codebase quickly.

---

## Product

- **Name:** CodexVid AI (Python package: `app`, version `1.0.0`)
- **Purpose:** Transform educational videos into interactive lessons. Upload a video (or YouTube URL) → transcribe with word-level timestamps → build sentence timeline → semantic chunk (30–60s) → embed in FAISS → two-stage LLM chat with sentence-level video seek → teaching pack (topics, takeaways, quiz)
- **Stack:** FastAPI + Uvicorn, faster-whisper, FAISS-CPU, Ollama/OpenAI/Anthropic/CompanyGPT, FFmpeg, yt-dlp, vanilla HTML/JS/CSS frontend

---

## What Is NOT in This Build

Older documentation or branches may reference these — they do not exist in this codebase:

- Workspace API or workspace sessions
- ChromaDB vector store
- SQLite or any relational database
- Job queue system
- User authentication / JWT
- Scoring pipeline
- Large SPA framework (React, Vue, etc.)

---

## Entry Points

| Entry Point | How to Use | What It Does |
|------------|-----------|--------------|
| `make dev` | `cd video-content-ai && make dev` | Start dev server with hot reload on port 8501 |
| `python -m app.cli serve` | `python -m app.cli serve [--host HOST] [--port PORT] [--reload]` | Production-style server launch |
| `python -m app.main` | Direct uvicorn run | Alternative to CLI |
| `make test` | `cd video-content-ai && make test` | Run all 28 pytest tests |
| `make lint` | `cd video-content-ai && make lint` | Run ruff linter on `app/` |
| `make clean` | `cd video-content-ai && make clean` | Remove `__pycache__`, `.egg-info`, etc. |
| `docker compose up --build` | From `video-content-ai/` | Start app + Ollama via Docker |

---

## Configuration

- **`app/config.py`** — All constants, parsed from `VCAI_*` environment variables at import time
- **`.env.example`** — Template; copy to `.env` for local development
- **No config object is passed around** — modules import constants directly from `app.config`
- **Precedence:** OS env vars > `.env` file > hardcoded defaults in `config.py`

Key config groups:
- **Server:** `VCAI_HOST`, `VCAI_PORT`, `VCAI_RELOAD`
- **Paths:** `VCAI_DATA_DIR`, `VCAI_CODEXVID_SESSIONS_DIR`
- **LLM:** `VCAI_LLM_PROVIDER`, `VCAI_DEFAULT_MODEL`, `VCAI_EMBEDDING_MODEL`, provider API keys
- **Whisper:** `VCAI_WHISPER_MODEL`, `VCAI_LANGUAGE`, `VCAI_STT_PROVIDER`
- **Pipeline tuning:** `VCAI_CODEXVID_*` (chunk sizes, overlap, parallel workers, top-k)

---

## File Map

```
video-content-ai/
├── app/
│   ├── __init__.py                 version = "1.0.0"
│   ├── main.py                     FastAPI app, middleware, route mounting, static serving
│   ├── config.py                   ALL config constants (VCAI_* env vars)
│   ├── cli.py                      Typer CLI: `serve` subcommand
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py               GET /health  GET /ready
│   │   └── codexvid.py             POST /upload  POST /chat  GET /exists  GET /video
│   │                               Chat body accepts segment_start/segment_end for FAISS time-range filtering
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── llm.py                  LLMProvider base + Ollama, OpenAI, Anthropic, CompanyGPT
│   │
│   ├── codexvid/
│   │   ├── __init__.py             Lazy imports
│   │   ├── session.py              new_session_dir(), process_upload(), load_store()
│   │   ├── transcription.py        transcribe_video() — parallel Whisper + window merging
│   │   ├── timestamp_utils.py      flatten_words, words_to_sentences, merge_segments, normalize
│   │   ├── chunking.py             create_chunks() — semantic 30–60s sentence-respecting chunks
│   │   ├── vector_store.py         CodexvidVectorStore — FAISS IndexFlatIP + metadata
│   │   ├── retrieval_utils.py      embed_texts, cosine_similarity, sentence filtering + pick
│   │   ├── chat.py                 detect_mode, chat() — two-stage LLM + grounding + timestamps
│   │   └── teaching.py             generate_teaching_output() — per-chunk LLM, merge, takeaways, quiz
│   │                               _extract_json(): 2-strategy robust JSON extractor (fence strip + block scan)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── transcription.py        Whisper model cache, audio splitting, language parsing
│   │   ├── video.py                yt-dlp download + normalize_media_source (5 functions only)
│   │   └── aws_transcribe.py       S3 upload, Transcribe job lifecycle, JSON parsing
│   │
│   └── static/
│       ├── learn.html              Three-screen SPA: upload → processing → workspace
│       │                           Includes #segment-context banner for active chapter display
│       ├── learn.js                Upload handler, chat, rendering, video seek (vanilla JS)
│       │                           Chapter interactions: timestamp seek, "Ask about this" auto-explain,
│       │                           segmentCache, segmentContext, active chapter tracking via timeupdate
│       └── learn.css               Dark theme, animations, responsive layout
│                                   65/35 grid (1.85fr/1fr), sticky video panel, chapter active highlight,
│                                   chapter-time-btn pill, btn-segment-chat, segment-context banner,
│                                   Thinking… animated dots
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py                 Health, ready, HTML serving
│   ├── test_codexvid_upload_api.py Upload validation tests
│   ├── test_codexvid_chunking.py   create_chunks, detect_mode, timestamp parsing
│   ├── test_timestamp_utils.py     Word/sentence merging utilities
│   ├── test_retrieval_utils.py     Cosine similarity, sentence filtering, snap
│   ├── test_teaching_pipeline.py   Topic merge, coverage enforcement
│   └── test_aws_transcribe.py      AWS JSON parsing (mocked)
│
├── pyproject.toml                  Package metadata, deps, build config
├── requirements.txt                Pinned pip snapshot
├── Makefile                        dev, test, lint, docker, clean targets
├── Dockerfile                      Multi-stage: Python 3.12 + ffmpeg
├── docker-compose.yml              App + Ollama service
├── .env.example                    VCAI_* variable template
├── .gitignore                      Ignores data/, .venv, __pycache__, etc.
├── .python-version                 pyenv version pin
└── *.md                            Documentation files
```

---

## Data Flow at a Glance

```
[Client] → POST /upload → [process_upload()]
                              ├─ FFmpeg → WAV
                              ├─ Whisper (parallel) → segments + words
                              ├─ timestamp_utils → sentences
                              ├─ chunking → 30–60s chunks
                              ├─ FAISS.build → faiss.index
                              └─ teaching → topics + quiz
                          → returns session_id + teaching

[Client] → POST /chat → [FAISS.search(k=3)]
                            └─ [chat()]
                                  ├─ sentence pick (cosine similarity)
                                  ├─ Stage 1: LLM extraction
                                  ├─ Stage 2: LLM explanation → JSON
                                  └─ grounding check
                        → returns answer + timestamp_start
```

---

## Key Architectural Decisions

1. **No database** — all session state stored as JSON + FAISS on disk; sessions identified by UUID
2. **Semantic chunking** — chunks follow sentence boundaries at 30–60s; never splits mid-sentence; time-based fallback guarantees multiple chunks even when Whisper word timestamps are sparse
3. **Forced sentence breaks** — `words_to_sentence_spans` forces a new sentence every 45 s to prevent one mega-sentence collapsing the entire transcript into a single chunk
4. **Sentence-level timestamps** — chat responses refined beyond chunk granularity to best-matching sentence
5. **Two-stage chat** — extract first (no summarization), then explain; prevents hallucination
6. **Per-chunk teaching** — one LLM call per 30–60s chunk, parallelized; prompt forbids whole-video summaries; post-response phrase detection substitutes raw snippet if LLM ignores the constraint
7. **Pluggable LLM** — swap backends by changing `VCAI_LLM_PROVIDER`; same interface for all
8. **FAISS cosine via normalization** — `IndexFlatIP` on L2-normalized vectors = exact cosine similarity
9. **Overlapping audio windows** — 5s overlap reduces Whisper boundary word drops
10. **Grounding check** — token overlap between answer and transcript; low confidence → safe fallback
11. **Thread pool for blocking I/O** — Whisper and LLM calls run in `ThreadPoolExecutor` so FastAPI event loop stays responsive
12. **Segment-scoped FAISS filtering** — `POST /api/codexvid/chat` accepts optional `segment_start`/`segment_end` floats; after FAISS retrieval, hits outside the requested time range are discarded so chat context stays within the user's selected chapter
13. **Auto-explain on segment click** — clicking "Ask about this" immediately triggers a `generateSegmentExplanation()` call without requiring user input; a "Thinking…" animated bubble appears while the API is in flight; result is cached in `segmentCache[idx]` for instant replay on re-click
14. **Stale-closure prevention** — the async chat submit handler captures `const activeSeg = segmentContext` synchronously before any `await`; all timestamp metadata uses the captured value, never the (potentially changed) module-level reference
15. **Robust LLM JSON extraction** — `_extract_json(raw)` in `teaching.py` uses a two-strategy fallback: (1) strip markdown fences then `json.loads`; (2) scan for outermost `{…}` block and parse that; handles Ollama models that prepend/append freeform text around JSON output

---

## Dependencies (Key)

| Package | Role |
|---------|------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `pydantic` | Request/response models |
| `faster-whisper` | Local STT with word timestamps |
| `faiss-cpu` | Vector similarity search |
| `numpy` | Vector math (L2 normalize, cosine) |
| `httpx` | HTTP client for LLM API calls |
| `yt-dlp` | YouTube video download |
| `ffmpeg` (system) | Audio extraction, format conversion |
| `boto3` | AWS S3 + Transcribe (optional) |
| `typer` | CLI framework |
| `difflib` | Title similarity for topic merging |

See `pyproject.toml` for full pinned list.

---

## Conventions

- **Config access:** Import directly from `app.config`, e.g. `from app.config import DEFAULT_MODEL`
- **LLM calls:** Always go through `app.core.llm.get_provider()` — never call provider SDKs directly
- **Session paths:** Always via `app.codexvid.session` helpers — never construct paths manually
- **Async vs sync:** FastAPI handlers are `async`; heavy work (Whisper, FAISS, LLM) runs in `run_in_executor`
- **Tests:** Use `pytest`; mock LLM calls and FAISS where integration is not needed; `TestClient` for HTTP

---

## Documentation Index

| Doc | Read When You Need To... |
|-----|--------------------------|
| [README.md](./README.md) | Get started, run the app, understand features and API |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Understand component boundaries and module roles |
| [APP_DATAFLOW.md](./APP_DATAFLOW.md) | Trace a request step-by-step through the pipeline |
| [TESTING.md](./TESTING.md) | Run tests, understand test coverage |
| [UI_CLICK_GUIDE.md](./UI_CLICK_GUIDE.md) | Understand what each UI element does and which API it calls |
