# Testing (CodexVid AI)

The suite is **pytest**-based and covers the CodexVid API, chunking helpers, timestamp utilities, retrieval utilities, teaching pipeline logic, and AWS Transcribe parsing. Tests use `TestClient` for HTTP tests and mocks to avoid real Whisper, FAISS, LLM, or AWS calls where the test only needs to verify logic or HTTP behavior.

**Current size:** 28 tests across 7 files.

---

## Running Tests

### Prerequisites

```bash
cd video-content-ai
pip install -e ".[dev]"    # installs pytest, pytest-asyncio, httpx, and app deps
```

### Run All Tests

```bash
make test
# equivalent to:
pytest tests/ -v
```

### Run a Single Test File

```bash
pytest tests/test_api.py -v
pytest tests/test_codexvid_upload_api.py -v
pytest tests/test_codexvid_chunking.py -v
pytest tests/test_timestamp_utils.py -v
pytest tests/test_retrieval_utils.py -v
pytest tests/test_teaching_pipeline.py -v
pytest tests/test_aws_transcribe.py -v
```

### Run a Single Test by Name

```bash
pytest tests/test_codexvid_chunking.py::test_create_chunks_returns_correct_shape -v
```

### Count Tests Without Running

```bash
pytest tests/ --collect-only -q
```

---

## Test Files

### `test_api.py` — 5 tests

Covers the core HTTP surface of the FastAPI app.

| Test | What It Checks |
|------|----------------|
| `test_health` | `GET /health` returns 200 with `{"status": "ok"}` and version fields |
| `test_ready_ok` | `GET /ready` returns 200 when LLM provider responds (mocked `list_models`) |
| `test_ready_fail` | `GET /ready` returns 503 when LLM provider raises an exception |
| `test_index_serves_learn_html` | `GET /` serves `learn.html` with CodexVid branding in body |
| `test_learn_serves_learn_html` | `GET /learn` serves same `learn.html` |

**Note:** There is no workspace API, ChromaDB, or scoring pipeline in this build — only the CodexVid routes listed above.

**Mocking pattern:**
```python
with patch("app.core.llm.get_provider") as mock_provider:
    mock_provider.return_value.list_models.return_value = ["llama3"]
    response = client.get("/ready")
```

---

### `test_codexvid_upload_api.py` — 3 tests

Covers upload validation and the YouTube path.

| Test | What It Checks |
|------|----------------|
| `test_upload_requires_file_or_url` | `POST /api/codexvid/upload` with neither `file` nor `youtube_url` → 422 |
| `test_upload_rejects_non_youtube_url` | `youtube_url` set to a non-YouTube URL (e.g. `https://example.com`) → 422 with validation error |
| `test_upload_youtube_success` | Valid YouTube URL with all pipeline steps mocked → 200 with `session_id` and `teaching` fields |

**Mocking pattern (YouTube success):**
```python
with patch("app.services.video.download_video") as mock_dl, \
     patch("app.codexvid.session.process_upload") as mock_proc:
    mock_dl.return_value = Path("/tmp/video.mp4")
    mock_proc.return_value = ("session_abc", {"segment_count": 5, ...})
    response = client.post("/api/codexvid/upload", data={"youtube_url": "https://youtube.com/watch?v=abc"})
```

---

### `test_codexvid_chunking.py` — 6 tests

Covers semantic chunking and chat utility functions.

| Test | What It Checks |
|------|----------------|
| `test_create_chunks_returns_correct_shape` | `create_chunks()` returns list of dicts with `text`, `start_time`, `end_time`, `start`, `end` keys |
| `test_create_chunks_respects_boundaries` | Output chunks never span longer than `SEM_CHUNK_MAX_SEC` seconds |
| `test_create_chunks_single_sentence` | Single-sentence input produces exactly one chunk |
| `test_detect_mode_simple` | `detect_mode("What is X?")` → `"simple"` |
| `test_mmss_label_to_seconds` | `mmss_label_to_seconds("01:23")` → `83.0` |
| `test_extract_timestamp_spans` | `extract_timestamp_spans("📍 01:20 – 02:00")` → list with `start_sec=80.0`, `end_sec=120.0` |

---

### `test_timestamp_utils.py` — 5 tests

Covers word/sentence processing utilities.

| Test | What It Checks |
|------|----------------|
| `test_merge_segments_deduplicates` | `merge_segments()` removes duplicate segments and enforces chronological order |
| `test_dedupe_overlapping_words` | Words with duplicate `start` timestamps are deduplicated (keeps first occurrence) |
| `test_words_to_fine_segments` | `words_to_fine_segments()` groups words into segments within `min_sec`–`max_sec` bounds |
| `test_normalize_transcript_segments` | `normalize_transcript_segments()` applies merge + align + clean in sequence |
| `test_clean_timestamps` | `align_timestamps()` snaps segment `start`/`end` to actual word timings |

---

### `test_retrieval_utils.py` — 4 tests

Covers the sentence similarity and retrieval helpers.

| Test | What It Checks |
|------|----------------|
| `test_cosine_similarity_matrix` | L2-normalized vectors yield similarity 1.0 with themselves, < 1.0 with others |
| `test_filter_sentences_overlapping_chunks` | Only sentences whose `[start, end]` overlaps a chunk's `[start_time, end_time]` are kept |
| `test_find_most_relevant_sentence` | With mocked `embed_texts`, returns sentence with highest cosine score to query |
| `test_snap_chapter_times_to_sentences` | Chapter boundaries snap to nearest sentence `start`/`end` timestamp |

**Mocking pattern:**
```python
with patch("app.codexvid.retrieval_utils.embed_texts") as mock_embed:
    mock_embed.return_value = np.array([[1, 0], [0, 1], [1, 0]])  # query, s1, s2
    result = find_most_relevant_sentence("query", sentences, model="nomic-embed-text")
    assert result == sentences[0]  # s1 has cosine 1.0 with query
```

---

### `test_teaching_pipeline.py` — 2 tests

Covers teaching pack post-processing (no LLM calls needed).

| Test | What It Checks |
|------|----------------|
| `test_merge_adjacent_topics` | Topics with ≥0.90 title similarity are merged into one; dissimilar topics stay separate |
| `test_enforce_coverage` | First topic's `start_time` becomes 0.0 (or first chunk start); last topic's `end_time` becomes last chunk end |

---

### `test_aws_transcribe.py` — 3 tests

Covers AWS Transcribe output parsing (no real AWS calls).

| Test | What It Checks |
|------|----------------|
| `test_parse_transcript_json_to_segments` | Full AWS Transcribe JSON → list of Whisper-compatible segments with `text`, `start`, `end`, `words` |
| `test_parse_transcript_json_empty_items` | AWS JSON with empty `items` array → returns empty segment list |
| `test_transcribe_path_to_segments_mocked` | `transcribe_path_to_segments()` with mocked boto3 → correct segment shape returned without real S3/job |

---

## Notes on Test Infrastructure

- **`TestClient`** is used for all HTTP tests — no separate server process needed
- **No real LLM calls** in tests — mocked wherever chat, embed, or teaching is involved
- **No real Whisper** — mocked at the service layer for upload tests
- **No real FAISS** — mocked for upload tests; `test_retrieval_utils` tests numpy math directly
- **No real AWS** — boto3 mocked in all `test_aws_transcribe.py` tests
- **For `/ready` without mocking**, your configured LLM endpoint must be running and respond to `list_models()` — this is the only test that hits a real external service (only if not mocked)

---

## Adding New Tests

1. Create `tests/test_<feature>.py`
2. Import `from fastapi.testclient import TestClient` and `from app.main import app`
3. Use `unittest.mock.patch` to mock external calls (LLM, Whisper, AWS, FAISS)
4. Test pure logic functions (e.g. `timestamp_utils`, `chunking`) without mocks — they are pure Python

---

See [README.md](./README.md) for runtime configuration and [ARCHITECTURE.md](./ARCHITECTURE.md) for module roles.
