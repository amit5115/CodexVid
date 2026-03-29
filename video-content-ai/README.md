# CodexVid AI

**CodexVid AI** is a FastAPI app for **video-grounded learning**: upload a video, get a **word-level** transcript (local Whisper or AWS Transcribe), build **sentence-level** timing, split into **semantic chunks** (≈30–60s, sentence boundaries), embed with **FAISS**, then **refine** chat seek times to the **best-matching sentence** inside retrieved chunks (cosine similarity on embeddings). Chat uses a **two-stage LLM** (extract → explain) with **grounding checks** and **float-accurate** seek timestamps. The **teaching pack** uses **per-chunk LLM** calls (one topic per semantic segment, no full-transcript summarization), then merge/coverage + optional **sentence snap**; takeaways/quiz come from a short **topic-summary** follow-up call.

There is **no** legacy workspace API, jobs database, ChromaDB, or scoring pipeline—only CodexVid sessions and the APIs below.

## Features

- **Video upload** — `POST /api/codexvid/upload` (multipart: `file` or YouTube URL)
- **Transcription** — `faster-whisper` locally or **AWS Transcribe** when `VCAI_STT_PROVIDER=aws`
- **Chunking + FAISS** — **semantic** chunks (topic-sized, not fixed word windows), each with `text`, `start_time`, `end_time` (precise seconds); vectors under `data/codexvid_sessions/<session_id>/` by default
- **CodexVid chat** — `POST /api/codexvid/chat` with RAG (**top‑k** chunks only, default **3**), then **sentence-level** timestamp pick from `transcript.json` + batch embeddings (`retrieval_utils`); multi-stage LLM; JSON fields (`timestamp_start`, `timestamp_end`, `key_points`, `grounded`, `grounding_score`)
- **Health** — `GET /health`, `GET /ready` (LLM reachability)

## Quick start

```bash
cd video-content-ai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env
# Set VCAI_LLM_BASE_URL / model, paths, and optional AWS vars if using Transcribe.

make dev
# → http://127.0.0.1:8501/   (learn UI; override with VCAI_PORT)
# → http://127.0.0.1:8501/docs
```

## Configuration

See **`.env.example`** and **`app/config.py`**. Important groups:

| Area | Notes |
|------|--------|
| Server | `VCAI_HOST`, `VCAI_PORT`, `VCAI_STATIC_DIR` |
| Paths | `VCAI_DATA_DIR`, `VCAI_CODEXVID_SESSIONS_DIR` (optional override for session storage root) |
| LLM | `VCAI_LLM_BASE_URL`, `VCAI_LLM_MODEL`, provider API keys if needed |
| Embeddings | `VCAI_EMBEDDING_MODEL` (sentence-vs-query similarity in chat; same as FAISS chunk embeddings) |
| Whisper | `VCAI_WHISPER_MODEL`, `VCAI_WHISPER_DEVICE`, `VCAI_WHISPER_COMPUTE_TYPE` |
| CodexVid pipeline | `VCAI_CODEXVID_CHUNK_SEC`, `VCAI_CODEXVID_AUDIO_OVERLAP_SEC`, `VCAI_CODEXVID_PARALLEL_WORKERS` (Whisper slicing); `VCAI_CODEXVID_FINE_SEG_*` (fine segments from words); **`VCAI_CODEXVID_RAG_TOP_K`** (default **3**, retrieval only); **`VCAI_CODEXVID_SEM_CHUNK_MIN_SEC`** / **`VCAI_CODEXVID_SEM_CHUNK_MAX_SEC`** (semantic chunk span, default **30** / **60**); legacy `VCAI_TEACHER_*` still read if new vars unset |
| STT | `VCAI_STT_PROVIDER` (`whisper` \| `aws`), AWS bucket/region/credentials for Transcribe |

## Project layout (high level)

```
app/
  main.py              # FastAPI: health + codexvid routes, static, / and /learn
  api/health.py        # GET /health, GET /ready
  api/codexvid.py      # upload, chat, session video, exists
  codexvid/            # chunking, timestamp_utils, retrieval_utils, chat (multi-stage), teaching, FAISS, session
  services/            # transcription, video, aws_transcribe
  core/llm.py          # LLM client (OpenAI-compatible + helpers)
  static/              # learn.html, learn.js, learn.css, index.html
tests/                 # pytest (see TESTING.md)
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Components and data flow |
| [APP_DATAFLOW.md](./APP_DATAFLOW.md) | Request/response paths |
| [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) | Repo map and conventions |
| [TESTING.md](./TESTING.md) | How to run tests |
| [UI_CLICK_GUIDE.md](./UI_CLICK_GUIDE.md) | Learn UI walkthrough |

## CLI

```bash
python -m app.cli serve [--host HOST] [--port PORT] [--reload]
```

## License

See repository root for license terms.
