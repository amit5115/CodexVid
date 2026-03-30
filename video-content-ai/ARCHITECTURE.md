# Architecture (CodexVid AI)

## System Overview

```
┌───────────────────────────────────────────────────────────────┐
│  Browser / API Client                                         │
│  • learn.html (upload, chat, lesson, video seek)              │
│  • Any HTTP client (curl, Postman, etc.)                      │
└───────────────────────────┬───────────────────────────────────┘
                            │  HTTP  (port 8501)
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  FastAPI  (app/main.py)                                        │
│  • CORS middleware                                             │
│  • Static files → /static/*                                    │
│  • / and /learn → learn.html                                   │
│  • /health, /ready                                             │
│  • /api/codexvid/*  (upload, chat, session video, exists)      │
└───────────┬─────────────────────────────────┬─────────────────┘
            │                                 │
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────────┐
│  app/codexvid/        │         │  app/services/             │
│  ─────────────        │         │  ───────────               │
│  session.py           │◄───────►│  video.py                  │
│  transcription.py     │         │  (FFmpeg, yt-dlp)          │
│  timestamp_utils.py   │         │                            │
│  chunking.py          │         │  transcription.py          │
│  vector_store.py      │         │  (Whisper helpers)         │
│  retrieval_utils.py   │         │                            │
│  chat.py              │         │  aws_transcribe.py         │
│  teaching.py          │         │  (S3, Transcribe job)      │
└───────────┬───────────┘         └───────────────────────────┘
            │                                 │
            │ embed / chat calls              │
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────────┐
│  app/core/llm.py       │         │  Disk: data/               │
│  ─────────────         │         │  ─────────────────         │
│  LLMProvider           │         │  codexvid_sessions/        │
│  • OllamaProvider      │         │    {session_id}/           │
│  • OpenAIProvider      │         │      source.mp4            │
│  • AnthropicProvider   │         │      transcript.json       │
│  • CompanyGPTProvider  │         │      chunks.json           │
│  chat(), embed(),      │         │      faiss.index           │
│  list_models()         │         │      faiss_meta.json       │
└───────────┬───────────┘         │      teaching.json         │
            │                     └───────────────────────────┘
            ▼
┌───────────────────────────────────────────────────────────────┐
│  LLM Backend (one of):                                         │
│  • Ollama   http://localhost:11434  (default)                  │
│  • OpenAI   api.openai.com                                     │
│  • Anthropic api.anthropic.com                                 │
│  • Company GPT internal proxy                                  │
└───────────────────────────────────────────────────────────────┘
```

---

## HTTP Surface

| Route | Method | Role |
|-------|--------|------|
| `/` | GET | Serve `learn.html` (no-cache) |
| `/learn` | GET | Alias for `/` |
| `/static/*` | GET | CSS, JS, assets |
| `/health` | GET | Liveness probe → `{"status": "ok"}` |
| `/ready` | GET | LLM reachability → 200 or 503 |
| `/api/codexvid/upload` | POST | Full upload pipeline |
| `/api/codexvid/chat` | POST | RAG chat with sentence timestamps |
| `/api/codexvid/sessions/{id}/exists` | GET | Index existence check |
| `/api/codexvid/sessions/{id}/video` | GET | Stream session video |
| `/docs` | GET | OpenAPI UI (auto-generated) |

No workspace router, no ChromaDB, no SQLite job queue in this build.

---

## Module Responsibilities

### `app/main.py`
- Creates the FastAPI application instance
- Adds CORS middleware (all origins in dev)
- Mounts the static files directory at `/static`
- Registers `/health`, `/ready` from `api/health.py`
- Registers all `/api/codexvid/*` routes from `api/codexvid.py`
- Serves `learn.html` for both `/` and `/learn` with `no-cache` headers

### `app/config.py`
- Single source of truth for all configuration constants
- Parses `VCAI_*` environment variables at import time (reads `.env` file if present)
- Exports typed constants used throughout the app (e.g. `CODEXVID_SESSIONS_DIR`, `DEFAULT_MODEL`, `RAG_TOP_K`)

### `app/cli.py`
- Typer-based CLI with a `serve` subcommand
- Arguments: `--host`, `--port`, `--reload`
- Calls `uvicorn.run(app)` directly

### `app/api/health.py`
- `GET /health` — always returns 200 with version/product fields
- `GET /ready` — calls `get_provider().list_models()` to verify the LLM backend is reachable; returns 503 if not

### `app/api/codexvid.py`
- Request/response Pydantic models for upload and chat
- `POST /api/codexvid/upload`: validates form, delegates to `process_upload()` in a thread pool executor (so async FastAPI isn't blocked), returns session metadata + teaching pack
- `POST /api/codexvid/chat`: loads FAISS store, runs FAISS search, **optionally filters hits to a specific time window** (`segment_start`/`segment_end` fields in request body — used by the UI when the user clicks "Ask about this" on a chapter), calls `chat()` in thread pool, returns structured JSON
  - `CodexvidChatBody` fields: `session_id`, `query`, `model`, `mode`, `segment_start` (optional float), `segment_end` (optional float)
  - If `segment_start`/`segment_end` are provided, FAISS hits outside `[segment_start, segment_end]` are filtered out; falls back to all hits if none overlap the window
- `GET /api/codexvid/sessions/{id}/exists`: calls `load_store()` and checks if index file exists
- `GET /api/codexvid/sessions/{id}/video`: streams `source.mp4` from the session directory

### `app/core/llm.py`
LLM provider abstraction with four backends:

| Class | Backend | Notes |
|-------|---------|-------|
| `OllamaProvider` | Local Ollama | Default; uses `http://localhost:11434` |
| `OpenAIProvider` | OpenAI API | Requires `OPENAI_API_KEY` |
| `AnthropicProvider` | Anthropic API | Requires `ANTHROPIC_API_KEY` |
| `CompanyGPTProvider` | Internal proxy | Auto-selected for gpt-4.1/gpt-4o/gpt-4o-mini model IDs |

All providers implement:
- `chat(messages, model, ...)` → str
- `chat_stream(messages, model, ...)` → AsyncGenerator
- `embed(model, texts)` → list[list[float]]
- `list_models()` → list[str]

`get_provider(model=None)` returns a cached singleton, routing company GPT models automatically.

### `app/codexvid/session.py`
- `new_session_dir()` → creates a UUID-based directory under `CODEXVID_SESSIONS_DIR`
- `process_upload(video_path, whisper_model, language, llm_model)` → orchestrates the entire pipeline:
  1. Copy video to session dir as `source.mp4`
  2. Transcribe with word-level timestamps
  3. Build sentence timeline
  4. Save `transcript.json`
  5. Create semantic chunks → save `chunks.json`
  6. Build and save FAISS index
  7. Generate and save teaching pack
- `load_store(session_id)` → loads `CodexvidVectorStore` from disk

### `app/codexvid/transcription.py`
- `transcribe_video(video_path, model_size, language)` → `list[segment_dict]`
  - Extracts 16 kHz mono WAV via FFmpeg
  - Splits audio into overlapping windows (configurable size + overlap)
  - Runs faster-whisper on each window in parallel (configurable workers)
  - Merges and deduplicates words across windows
  - Returns segments with `text`, `start`, `end`, `words: [{word, start, end}]`

### `app/codexvid/timestamp_utils.py`
- `flatten_words_from_transcript(segments)` → flat word list with timestamps (interpolates if missing)
- `words_to_sentence_spans(words)` → groups words into sentences by punctuation; **forces a sentence break every 45 s** (`_MAX_SENTENCE_SEC = 45.0`) to prevent a single mega-sentence when Whisper omits punctuation; each span has `text`, `start`, `end`, `words`
- `transcript_sentence_timeline(segments)` → full sentence list without word details
- `merge_segments(segments)` → deduplicates and enforces chronological order; **substring containment only applied to short stubs (< 4 words)** to prevent cascading segment collapse
- `words_to_fine_segments(words, min_sec, max_sec)` → short 2–5s segments
- `normalize_transcript_segments(segments)` → merge + align + clean
- `align_timestamps(segments)` → snaps to actual word timings, enforces monotonic boundaries

### `app/codexvid/chunking.py`
- `create_chunks(transcript, ...)` → `list[chunk_dict]`
  - Builds sentences from word timings via `timestamp_utils`
  - Greedily packs sentences into chunks of `SEM_CHUNK_MIN_SEC`–`SEM_CHUNK_MAX_SEC` (default 30–60s)
  - Splits oversized sentences at word boundaries; **if a sentence has no word timestamps, produces a single covering chunk instead of silently dropping it**
  - **Post-validation:** if word-based chunking produces fewer chunks than `floor(video_duration / max_sec)`, switches to `_chunk_segments_by_time()` time-based fallback to guarantee multiple chunks
  - Each chunk: `{"text", "start_time", "end_time", "start", "end"}`
- `_chunk_segments_by_time(segments, min_sec, max_sec)` → direct time-based chunking fallback (no word timestamp dependency)
- `_expected_min_chunks(video_duration_sec, max_sec)` → minimum chunk count for a given video duration
- `chunk_time_range(chunk)` → `(start_sec, end_sec)` tuple

### `app/codexvid/vector_store.py`
- `CodexvidVectorStore` class:
  - `index`: `faiss.IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
  - `meta`: list of chunk dicts with `text`, `start_time`, `end_time`
  - `build(chunks, session_dir, embed_fn)` → embeds all chunks, L2-normalizes, adds to FAISS, saves files
  - `search(query, k)` → embeds query, L2-normalizes, returns top-k chunks with scores
  - `save()` / `load(session_dir)` → serialize/deserialize `faiss.index` + `faiss_meta.json`
- Embedding uses `get_provider().embed(model=EMBEDDING_MODEL, texts=[...])`

### `app/codexvid/retrieval_utils.py`
- `embed_texts(texts, model)` → `np.ndarray (n, dim)` — single batch embedding call
- `cosine_similarity_matrix(query_vec, matrix)` → similarity scores
- `filter_sentences_overlapping_chunks(sentences, chunks)` → keeps sentences that overlap any chunk's time range
- `find_most_relevant_sentence(query, sentences, embed_model)` → one batch embed of `[query] + sentences`, returns highest-scoring sentence
- `load_session_sentences(session_id)` → loads from `transcript.json` `sentences` field (or rebuilds from `segments` for legacy sessions)

### `app/codexvid/chat.py`
- `detect_mode(query)` → auto-detects `simple`, `detailed`, `analogy`, `example`, `default` from query keywords
- `chat(query, retrieved_chunks, model, mode, session_id)` → full pipeline:
  1. Sentence-level timestamp refinement via `retrieval_utils`
  2. Stage 1: extraction prompt → all relevant points (no summarization)
  3. Check empty extraction → return "not in video" early
  4. Stage 2: explanation prompt → JSON `{answer, timestamp_start, timestamp_end, key_points}`
  5. Grounding check: ≥38% of meaningful answer tokens must appear in transcript; otherwise return safe fallback
  6. Return dict with all fields
- `mmss_label_to_seconds(label)` → parses `"mm:ss"` or `"h:mm:ss"` to float
- `extract_timestamp_spans(text)` → legacy: extracts `📍 mm:ss` labels from answer text

### `app/codexvid/teaching.py`
- `generate_teaching_output(chunks, sentences, model, workers)` → full teaching pack:
  1. Parallel LLM calls (one per chunk) → `{topic_title, description, start_time, end_time}`; **system prompt explicitly forbids whole-video summaries and cross-segment references**
  2. **`_extract_json(raw)`** — robust JSON extractor used for all LLM responses; tries two strategies: (a) direct parse after markdown-fence stripping, (b) scan for outermost `{…}` block to handle preamble text or trailing notes from local LLMs; logs the first 120 chars of raw output on failure
  3. **Whole-video phrase detection:** if the LLM response contains phrases like "in this video", "throughout the video", or "the video covers", the raw transcript snippet is substituted to ensure segment-level accuracy
  4. `merge_adjacent_topics()` — merge consecutive topics with title similarity ≥ 0.90 (difflib ratio)
  5. `enforce_coverage()` — extend first/last topic to span full video; triggered earlier (`_LONG_VIDEO_SEC = 120.0`, down from 300.0)
  6. `snap_chapter_times_to_sentences()` — if sentences provided, snap chapter boundaries to nearest sentence
  7. Second LLM call on topic summaries → `{key_takeaways: [...], quiz: [{question, answer}]}`; also uses `_extract_json`
- Response shape: `{topics: [...], chapters: [...], key_takeaways: [...], quiz: [...]}`

### `app/services/transcription.py`
- Whisper model loading (cached singleton per model size)
- `_get_audio_duration(audio_path)` → via ffprobe
- `_split_audio(audio_path, chunk_duration, overlap_sec)` → list of `(path, offset)` tuples
- `_transcribe_one_chunk(chunk_path, model, language)` → segment list with word timings
- `_parse_language(language)` → handles auto-detect and optional translation (e.g. `"en→es"`)

### `app/services/video.py`
- `is_url(source)` → returns True if the source string has an http/https scheme
- `normalize_media_source(source)` → prepends `https://` to bare YouTube/video host strings missing a scheme
- `download_video(url, output_dir)` → downloads via yt-dlp with multi-format + cookie fallback, returns local MP4 path
- `_ydl_base_opts(use_cookies)` → builds shared yt-dlp options: socket timeout, iOS/Android player clients, optional browser cookies
- `_find_node()` → locates the Node.js binary for yt-dlp JS runtime (optional)

### `app/services/aws_transcribe.py`
- `transcribe_path_to_segments(audio_path, language, bucket, region, ...)` → Whisper-compatible segment list
  - Uploads audio to S3
  - Starts an AWS Transcribe batch job
  - Polls until completion or timeout
  - Parses Transcribe JSON response
- `parse_transcript_json_to_segments(aws_json)` → normalizes AWS output to match local Whisper format

---

## Data Storage

All session state is stored on disk — no database.

```
{VCAI_CODEXVID_SESSIONS_DIR}/{session_id}/
├── source.mp4          # Original uploaded or downloaded video
├── transcript.json     # {"segments": [...], "sentences": [...]}
├── chunks.json         # All semantic chunks with text + timestamps
├── faiss.index         # FAISS binary (IndexFlatIP, L2-normalized)
├── faiss_meta.json     # {"dim": int, "meta": [{text, start_time, end_time}]}
└── teaching.json       # {topics, chapters, key_takeaways, quiz}
```

**`transcript.json` schema:**
```json
{
  "segments": [
    {
      "text": "Hello world.",
      "start": 0.0,
      "end": 2.5,
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world.", "start": 0.6, "end": 1.0}
      ]
    }
  ],
  "sentences": [
    {"text": "Hello world.", "start": 0.0, "end": 2.5}
  ]
}
```

---

## Configuration Flow

```
.env file
    └─► app/config.py (parsed at import time)
            └─► Constants used throughout modules
                    └─► Overridable per-request via API parameters (model, language, etc.)
```

---

## LLM Provider Selection

```
get_provider(model=None)
    ├─ model in ["gpt-4.1", "gpt-4o", "gpt-4o-mini"] → CompanyGPTProvider
    ├─ VCAI_LLM_PROVIDER=openai → OpenAIProvider
    ├─ VCAI_LLM_PROVIDER=anthropic → AnthropicProvider
    ├─ VCAI_LLM_PROVIDER=company_gpt → CompanyGPTProvider
    └─ default (ollama) → OllamaProvider
```

---

## Tests

`tests/` covers the CodexVid API, chunking, retrieval helpers, timestamp utilities, and AWS Transcribe parsing (mocked). See [TESTING.md](./TESTING.md) for full details.
