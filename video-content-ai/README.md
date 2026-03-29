# CodexVid AI

**CodexVid AI** is a FastAPI-based backend application for **video-grounded learning**. The core workflow is:

1. Upload a video file or provide a YouTube URL
2. Extract audio with FFmpeg, then transcribe with **word-level timestamps** (faster-whisper locally or AWS Transcribe in batch mode)
3. Build a **sentence-level timeline** from word data (punctuation-based grouping)
4. Split the transcript into **semantic chunks** (~30–60 seconds each, sentence-boundary-aware)
5. Embed each chunk and store vectors in **FAISS** for fast similarity search
6. On chat requests: retrieve top-k chunks, refine the answer timestamp to the **best-matching sentence** using cosine similarity on embeddings, then run a **two-stage LLM** (extract → explain) with grounding checks
7. Generate a structured **teaching pack** (topics/chapters, key takeaways, quiz) per-chunk using parallel LLM calls

There is **no** legacy workspace API, jobs database, ChromaDB, or scoring pipeline — only CodexVid sessions and the APIs below.

---

## Features

- **Video upload** — `POST /api/codexvid/upload` (multipart: `file` or `youtube_url`)
- **YouTube support** — automatic download via yt-dlp (with socket timeout and iOS/Android client fallback)
- **Transcription** — `faster-whisper` locally (word-level, overlapping audio windows) or **AWS Transcribe** batch when `VCAI_STT_PROVIDER=aws`
- **Semantic chunking** — topic-sized 30–60s windows respecting sentence boundaries (not fixed word windows)
- **FAISS vector store** — per-session `faiss.index` + `faiss_meta.json` with `start_time`/`end_time` per chunk
- **Sentence-level timestamp refinement** — one batched embedding call, cosine similarity picks the single best-matching sentence inside the retrieved chunks; exact float seconds returned for `HTMLMediaElement.currentTime` seeking
- **Two-stage grounded chat** — Stage 1: extract all relevant points; Stage 2: explain like a teacher → JSON with `answer`, `timestamp_start`, `timestamp_end`, `key_points`; grounding check prevents hallucination
- **Teaching pack** — one LLM call per semantic chunk (parallel), aggregate → merge adjacent similar topics → enforce video coverage → optional sentence-snap; takeaways + quiz from a follow-up summary call
- **Pluggable LLM** — Ollama (default), OpenAI, Anthropic, or internal Company GPT proxy
- **Health endpoints** — `GET /health` (liveness), `GET /ready` (LLM reachability)
- **Premium UI** — dark-mode SPA (`learn.html`) with video playback, Lesson/Chat tabs, and click-to-seek from chat timestamps

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
| | `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| | `VCAI_DEFAULT_MODEL` | `llama3` | Chat/teaching LLM |
| | `VCAI_VISION_MODEL` | `llava` | Vision model (optional) |
| | `OPENAI_API_KEY` | — | Required for OpenAI provider |
| | `ANTHROPIC_API_KEY` | — | Required for Anthropic provider |
| | `COMPANY_GPT_ENDPOINT` | — | Internal proxy URL |
| | `COMPANY_GPT_API_KEY` | — | Internal proxy key |
| | `COMPANY_GPT_CALLER` | — | Caller email for audit |
| | `COMPANY_GPT_VERIFY_SSL` | `true` | SSL verification |
| **Embeddings** | `VCAI_EMBEDDING_MODEL` | `nomic-embed-text` | Model for FAISS indexing and sentence similarity |
| **Whisper** | `VCAI_WHISPER_MODEL` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large` |
| | `VCAI_LANGUAGE` | `en` | Transcription language code |
| **STT** | `VCAI_STT_PROVIDER` | `whisper` | `whisper` or `aws` |
| | `AWS_REGION` | — | AWS region for Transcribe |
| | `AWS_ACCESS_KEY_ID` | — | AWS credentials |
| | `AWS_SECRET_ACCESS_KEY` | — | AWS credentials |
| | `VCAI_AWS_TRANSCRIBE_BUCKET` | — | S3 bucket for audio upload |
| | `VCAI_AWS_TRANSCRIBE_OUTPUT_PREFIX` | `transcripts/` | S3 key prefix for output |
| | `VCAI_AWS_TRANSCRIBE_POLL_TIMEOUT_SEC` | `3600` | Max polling time in seconds |
| **Pipeline** | `VCAI_CODEXVID_CHUNK_SEC` | `25` | Whisper audio window size (seconds) |
| | `VCAI_CODEXVID_AUDIO_OVERLAP_SEC` | `5` | Window overlap to reduce boundary drops |
| | `VCAI_CODEXVID_PARALLEL_WORKERS` | `4` | Parallel Whisper jobs |
| | `VCAI_CODEXVID_FINE_SEG_MIN_SEC` | `2` | Min fine segment duration (seconds) |
| | `VCAI_CODEXVID_FINE_SEG_MAX_SEC` | `5` | Max fine segment duration (seconds) |
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

Upload a video file or YouTube URL. Runs the full pipeline (transcribe → chunk → embed → teach) and returns the session ID plus teaching pack.

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
  "source": "upload",
  "youtube_url": null
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
  "mode": "simple"
}
```

**Response:**
```json
{
  "session_id": "abc123...",
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

Check whether a session's FAISS index exists on disk.

**Response:** `{ "session_id": "...", "exists": true }`

#### `GET /api/codexvid/sessions/{session_id}/video`

Stream the original video for playback. Returns the file with appropriate `Content-Type` (e.g. `video/mp4`).

---

## Project Layout

```
video-content-ai/
├── app/
│   ├── main.py                    # FastAPI app: middleware, route mounting, static serving
│   ├── config.py                  # All config constants; parsed from VCAI_* env vars
│   ├── cli.py                     # Typer CLI: `serve` command
│   ├── __init__.py                # Package version (1.0.0)
│   ├── api/
│   │   ├── health.py              # GET /health, GET /ready
│   │   └── codexvid.py            # POST /upload, /chat, GET /exists, /video
│   ├── core/
│   │   └── llm.py                 # LLM provider abstraction (Ollama, OpenAI, Anthropic, CompanyGPT)
│   ├── codexvid/
│   │   ├── session.py             # process_upload() orchestration; session I/O
│   │   ├── transcription.py       # transcribe_video(): parallel Whisper + overlapping windows
│   │   ├── chunking.py            # create_chunks(): semantic 30–60s sentence-respecting chunks
│   │   ├── timestamp_utils.py     # Words → sentences; merge, normalize, deduplicate
│   │   ├── vector_store.py        # CodexvidVectorStore: FAISS IndexFlatIP + metadata JSON
│   │   ├── chat.py                # Two-stage LLM chat: extract → explain; grounding; timestamps
│   │   ├── retrieval_utils.py     # Sentence filtering + cosine similarity for timestamp refinement
│   │   └── teaching.py            # Per-chunk LLM topics; merge, coverage, snap; takeaways + quiz
│   └── services/
│       ├── transcription.py       # Whisper model loading, audio splitting, language helpers
│       ├── video.py               # FFmpeg extraction, yt-dlp download, video metadata
│       └── aws_transcribe.py      # S3 upload, Transcribe job, polling, JSON parsing
├── app/static/
│   ├── learn.html                 # Main SPA: upload, processing, workspace screens
│   ├── learn.js                   # Vanilla JS: upload handler, chat, rendering, video seek
│   ├── learn.css                  # Dark theme UI (gradients, animations, responsive)
│   └── index.html                 # Landing page (currently unused)
├── tests/
│   ├── test_api.py
│   ├── test_codexvid_upload_api.py
│   ├── test_codexvid_chunking.py
│   ├── test_timestamp_utils.py
│   ├── test_retrieval_utils.py
│   ├── test_teaching_pipeline.py
│   └── test_aws_transcribe.py
├── pyproject.toml                 # Package metadata, dependencies, build config
├── requirements.txt               # Pinned pip snapshot
├── Makefile                       # make dev / test / lint / docker / clean
├── Dockerfile                     # Multi-stage: Python 3.12 + ffmpeg
├── docker-compose.yml             # App + Ollama service
├── .env.example                   # Template for VCAI_* variables
└── *.md                           # Documentation files
```

---

## Documentation Index

| Doc | Purpose |
|-----|---------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Component diagram, HTTP surface, module roles |
| [APP_DATAFLOW.md](./APP_DATAFLOW.md) | Step-by-step data flow for upload, transcription, chunking, chat, teaching |
| [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) | Repo map, conventions, entry points for contributors and AI assistants |
| [TESTING.md](./TESTING.md) | Test suite overview, how to run, what each file covers |
| [UI_CLICK_GUIDE.md](./UI_CLICK_GUIDE.md) | Walkthrough of every UI element in the learn interface |

---

## CLI

```bash
# Development (hot reload)
make dev

# Production
python -m app.cli serve [--host HOST] [--port PORT] [--reload]

# Tests
make test

# Lint
make lint

# Clean build artifacts
make clean
```

---

## License

See repository root for license terms.
