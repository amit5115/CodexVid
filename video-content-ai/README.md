# CodexVid AI

**CodexVid AI** is a FastAPI-based backend application for **video-grounded learning**. The core workflow is:

1. Upload a video file or provide a YouTube URL
2. Extract audio with FFmpeg, then transcribe with **word-level timestamps** (faster-whisper locally or AWS Transcribe in batch mode)
3. Build a **sentence-level timeline** from word data — with forced breaks every 45 s to prevent a single mega-sentence when Whisper omits punctuation
4. Split the transcript into **semantic chunks** (~30–60 seconds each, sentence-boundary-aware), with a robust time-based fallback and minimum-chunk-count enforcement
5. Embed each chunk and store vectors in **FAISS** for fast similarity search
6. On chat requests: retrieve top-k chunks, refine the answer timestamp to the **best-matching sentence** using cosine similarity on embeddings, then run a **two-stage LLM** (extract → explain) with grounding checks
7. Generate a structured **teaching pack** (topics/chapters, key takeaways, quiz) per-chunk using parallel LLM calls

There is **no** legacy workspace API, jobs database, ChromaDB, or scoring pipeline — only CodexVid sessions and the APIs below.

---

## Features

- **Video upload** — `POST /api/codexvid/upload` (multipart: `file` or `youtube_url`)
- **YouTube support** — automatic download via yt-dlp (with socket timeout and iOS/Android client fallback)
- **Transcription** — `faster-whisper` locally (word-level, overlapping audio windows) or **AWS Transcribe** batch when `VCAI_STT_PROVIDER=aws`
- **Robust semantic chunking** — 30–60 s topic windows with sentence boundaries; time-based fallback guarantees multiple chunks even when Whisper word timestamps are unavailable
- **FAISS vector store** — per-session `faiss.index` + `faiss_meta.json` with `start_time`/`end_time` per chunk
- **Sentence-level timestamp refinement** — one batched embedding call, cosine similarity picks the best-matching sentence inside the retrieved chunks; exact float seconds for `HTMLMediaElement.currentTime`
- **Two-stage grounded chat** — Stage 1: extract all relevant points; Stage 2: explain like a teacher → JSON with `answer`, `timestamp_start`, `timestamp_end`, `key_points`; grounding check prevents hallucination
- **Teaching pack** — one LLM call per semantic chunk (parallel), aggregate → merge adjacent similar topics → enforce video coverage → sentence-snap; takeaways + quiz from a follow-up summary call
- **Whole-video-summary detection** — per-chunk LLM responses are inspected for phrases like "in this video" or "throughout the video"; if found the raw transcript snippet is substituted to ensure segment-level accuracy
- **Pluggable LLM** — Ollama (default), OpenAI, Anthropic, or internal Company GPT proxy
- **Health endpoints** — `GET /health` (liveness), `GET /ready` (LLM reachability)
- **Premium UI** — dark-mode SPA (`learn.html`) with:
  - 65/35 workspace layout: sticky video panel on left, tabbed Lesson/Chat panel on right
  - **Chapter cards** — each shows a clickable timestamp pill (mm:ss format) that seeks the video, an active-highlight border when the video is playing through that segment, and an **"Ask about this"** button
  - **Auto-explain on "Ask about this"** — clicking the button immediately fires an API request without any user typing; an animated "Thinking…" bubble appears, then the explanation replaces it; result is cached per chapter for instant replay
  - **Segment context banner** — shows the active chapter while chatting about a specific segment; can be dismissed to return to whole-video chat
  - **Segment-scoped chat** — follow-up messages typed while a segment context is active are automatically scoped to that chapter's time range
  - **Timestamps in mm:ss format** — jump-to-segment links in chat display human-readable times (e.g. `7:03 – 7:16`) instead of raw seconds

---

## Quick Start

### Local Development

```bash
cd video-content-ai

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -e ".[dev]"

cp .env.example .env
# Edit .env as needed (LLM provider, model, paths, etc.)

make dev
# → http://127.0.0.1:8501/        (learn UI)
# → http://127.0.0.1:8501/docs    (OpenAPI explorer)
```

### Docker

```bash
docker compose up --build
# Starts app on port 8501 + Ollama service
```

### Production CLI

```bash
python -m app.cli serve --host 0.0.0.0 --port 8501 --no-reload
```

---

## Configuration

All configuration lives in **`app/config.py`** and is overridden by environment variables prefixed with `VCAI_*`. Copy `.env.example` to `.env` for local development.

| Area | Variable | Default | Description |
|------|----------|---------|-------------|
| **Server** | `VCAI_HOST` | `0.0.0.0` | Bind address |
| | `VCAI_PORT` | `8501` | HTTP port |
| | `VCAI_RELOAD` | `true` | Uvicorn hot reload |
| **Paths** | `VCAI_DATA_DIR` | `./data` | Root data directory |
| | `VCAI_CODEXVID_SESSIONS_DIR` | `{DATA_DIR}/codexvid_sessions` | Per-session artifacts |
| **LLM** | `VCAI_LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `anthropic`, `company_gpt` |
| | `VCAI_DEFAULT_MODEL` | `llama3` | Chat/teaching LLM |
| | `OPENAI_API_KEY` | — | Required for OpenAI provider |
| | `ANTHROPIC_API_KEY` | — | Required for Anthropic provider |
| | `COMPANY_GPT_ENDPOINT` | — | Internal proxy URL |
| | `COMPANY_GPT_API_KEY` | — | Internal proxy key |
| | `COMPANY_GPT_CALLER` | — | Caller identity for audit |
| **Embeddings** | `VCAI_EMBEDDING_MODEL` | `nomic-embed-text` | Model for FAISS indexing and sentence similarity |
| **STT** | `VCAI_STT_PROVIDER` | `whisper` | `whisper` or `aws` |
| | `AWS_REGION` | `us-east-1` | AWS region |
| | `AWS_ACCESS_KEY_ID` | — | AWS credentials |
| | `AWS_SECRET_ACCESS_KEY` | — | AWS credentials |
| | `VCAI_AWS_TRANSCRIBE_BUCKET` | — | S3 bucket for audio upload |
| | `VCAI_AWS_TRANSCRIBE_POLL_TIMEOUT_SEC` | `3600` | Max polling time |
| **Pipeline** | `VCAI_CODEXVID_CHUNK_SEC` | `25` | Whisper audio window size (seconds) |
| | `VCAI_CODEXVID_AUDIO_OVERLAP_SEC` | `5` | Window overlap to reduce boundary drops |
| | `VCAI_CODEXVID_PARALLEL_WORKERS` | `4` | Parallel Whisper jobs |
| | `VCAI_CODEXVID_FINE_SEG_MIN_SEC` | `2` | Min fine segment duration |
| | `VCAI_CODEXVID_FINE_SEG_MAX_SEC` | `5` | Max fine segment duration |
| | `VCAI_CODEXVID_RAG_TOP_K` | `3` | Top-k chunks retrieved per chat query |
| | `VCAI_CODEXVID_SEM_CHUNK_MIN_SEC` | `30` | Min semantic chunk duration (seconds) |
| | `VCAI_CODEXVID_SEM_CHUNK_MAX_SEC` | `60` | Max semantic chunk duration (seconds) |
| | `VCAI_CODEXVID_TEACHING_CHUNK_WORKERS` | `4` | Parallel LLM calls for teaching pack |
| **YouTube** | `VCAI_YTDLP_SOCKET_TIMEOUT` | `120` | yt-dlp socket timeout (seconds) |

---

## API Reference

### Health

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `GET` | `/health` | Liveness probe | `{"status": "ok", "version": "1.0.0", "product": "codexvid-ai"}` |
| `GET` | `/ready` | LLM reachability | `{"status": "ready"/"not_ready", "checks": {...}}` (200 or 503) |

### Static / UI

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve `learn.html` |
| `GET` | `/learn` | Alias for `/` |
| `GET` | `/static/*` | CSS, JS, assets |
| `GET` | `/docs` | OpenAPI UI |

### CodexVid

#### `POST /api/codexvid/upload`

Upload a video file or YouTube URL. Runs the full pipeline and returns session ID plus teaching pack.

**Form fields:**

| Field | Type | Description |
|-------|------|-------------|
| `file` | UploadFile | Video file (omit if using `youtube_url`) |
| `youtube_url` | str | YouTube URL (omit if uploading a file) |
| `whisper_model` | str | Whisper model size, e.g. `base` |
| `language` | str | Language code, e.g. `en` |
| `model` | str | LLM model for teaching, e.g. `llama3` |

**Response:**
```json
{
  "session_id": "abc123...",
  "segment_count": 42,
  "chunk_count": 8,
  "teaching": {
    "topics": [
      { "topic_title": "...", "description": "...", "start_time": 0.0, "end_time": 30.5 }
    ],
    "chapters": [...],
    "key_takeaways": ["..."],
    "quiz": [{ "question": "...?", "answer": "..." }]
  },
  "source": "upload"
}
```

#### `POST /api/codexvid/chat`

Ask a question about a session's video content.

**Request body:**
```json
{
  "session_id": "abc123...",
  "query": "What is this about?",
  "model": "llama3",
  "mode": "simple",
  "segment_start": 423.44,
  "segment_end": 486.0
}
```

`segment_start` and `segment_end` are optional floats (seconds). When provided, FAISS hits outside the specified time range are filtered out so the LLM context is restricted to that chapter. Used by the "Ask about this" feature to prevent cross-segment contamination.

**Response:**
```json
{
  "answer": "...",
  "timestamp_start": 12.345,
  "timestamp_end": 45.678,
  "key_points": ["point 1", "point 2"],
  "grounded": true,
  "grounding_score": 0.95,
  "mode": "simple",
  "chunks_used": 3,
  "timestamps": [
    { "start_label": "00:12", "end_label": "00:45", "start_sec": 12.345, "end_sec": 45.678 }
  ]
}
```

#### `GET /api/codexvid/sessions/{session_id}/exists`

Check whether a session's FAISS index exists. Returns `{ "exists": true/false }`.

#### `GET /api/codexvid/sessions/{session_id}/video`

Stream the original video for playback.

---

## Project Layout

```
video-content-ai/
├── app/
│   ├── main.py                    # FastAPI app: middleware, route mounting, static serving
│   ├── config.py                  # All config constants; parsed from VCAI_* env vars
│   ├── cli.py                     # Typer CLI: `serve` command
│   ├── api/
│   │   ├── health.py              # GET /health, GET /ready
│   │   └── codexvid.py            # POST /upload, /chat, GET /exists, /video
│   ├── core/
│   │   └── llm.py                 # LLM provider abstraction (Ollama, OpenAI, Anthropic, CompanyGPT)
│   ├── codexvid/
│   │   ├── session.py             # process_upload() orchestration; session I/O
│   │   ├── transcription.py       # transcribe_video(): parallel Whisper + overlapping windows
│   │   ├── chunking.py            # create_chunks(): semantic chunks with fallback enforcement
│   │   ├── timestamp_utils.py     # Words → sentences (with forced 45s breaks); merge/normalize
│   │   ├── vector_store.py        # CodexvidVectorStore: FAISS IndexFlatIP + metadata JSON
│   │   ├── chat.py                # Two-stage LLM chat: extract → explain; grounding; timestamps
│   │   ├── retrieval_utils.py     # Sentence filtering + cosine similarity for timestamp refinement
│   │   └── teaching.py            # Per-chunk LLM topics; merge, coverage, snap; takeaways + quiz
│   └── services/
│       ├── transcription.py       # Whisper model loading, audio splitting, language helpers
│       ├── video.py               # yt-dlp download + normalize_media_source
│       └── aws_transcribe.py      # S3 upload, Transcribe job, polling, JSON parsing
├── app/static/
│   ├── learn.html                 # Main SPA: upload, processing, workspace screens
│   ├── learn.js                   # Vanilla JS: upload handler, chat, rendering, video seek
│   └── learn.css                  # Dark theme UI (gradients, animations, responsive)
├── tests/                         # 28 pytest tests
├── pyproject.toml
├── requirements.txt
├── Makefile
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Documentation Index

| Doc | Purpose |
|-----|---------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Component diagram, HTTP surface, module roles |
| [APP_DATAFLOW.md](./APP_DATAFLOW.md) | Step-by-step data flow for every pipeline stage |
| [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) | Repo map, conventions, entry points |
| [TESTING.md](./TESTING.md) | Test suite overview, how to run |
| [UI_CLICK_GUIDE.md](./UI_CLICK_GUIDE.md) | Every UI element and which API it calls |

---

## CLI

```bash
make dev          # hot-reload dev server
make test         # run all 28 tests
make lint         # ruff linter
make clean        # remove build artifacts
python -m app.cli serve [--host HOST] [--port PORT] [--reload]
```

---

## License

See repository root for license terms.
