# Testing (CodexVid AI)

The suite is **pytest**-based and targets the CodexVid app: health/ready, learn UI HTML, codexvid upload validation, chunking helpers, **retrieval_utils** (sentence similarity helpers), transcript timestamp utilities, and AWS Transcribe parsing (mocked).

**Current size:** run `pytest tests/ --collect-only -q` — expect **28** tests.

## Run

```bash
cd video-content-ai
pip install -e ".[dev]"   # if not already

make test
# or
pytest tests/ -v
```

Run a single file:

```bash
pytest tests/test_api.py -v
pytest tests/test_codexvid_chunking.py -v
pytest tests/test_timestamp_utils.py -v
pytest tests/test_retrieval_utils.py -v
pytest tests/test_teaching_pipeline.py -v
```

## Test files

| File | Count | What it covers |
|------|------:|----------------|
| `test_api.py` | 5 | `GET /health`, `GET /ready` (mocked LLM), `GET /` and `/learn` serve learn UI with CodexVid branding, `POST /api/workspace/generate` returns **404** (legacy removed). |
| `test_codexvid_upload_api.py` | 3 | `POST /api/codexvid/upload`: requires file or YouTube URL; rejects non-YouTube URL; mocked YouTube path returns success. |
| `test_codexvid_chunking.py` | 6 | `create_chunks` (semantic chunks; `start_time`/`end_time`), `detect_mode`, `mmss_label_to_seconds`, `extract_timestamp_spans` from `app/codexvid/chat.py`. |
| `test_timestamp_utils.py` | 5 | `merge_segments`, `dedupe_overlapping_words`, `words_to_fine_segments`, `normalize_transcript_segments`, `clean_timestamps`. |
| `test_retrieval_utils.py` | 4 | `cosine_similarity_matrix`, `filter_sentences_overlapping_chunks`, `find_most_relevant_sentence` (mocked embed), `snap_chapter_times_to_sentences`. |
| `test_teaching_pipeline.py` | 2 | `merge_adjacent_topics`, `enforce_coverage` (no LLM). |
| `test_aws_transcribe.py` | 3 | Parse Transcribe JSON → segments; empty items; mocked `transcribe_path_to_segments` (no real AWS). |

## Notes

- **`TestClient`** is used; no separate server process.
- **Mocks** avoid real Whisper, FAISS, downloads, and boto3 where tests only need HTTP or pure logic.
- For `/ready` to pass without mocks, your LLM endpoint must respond to `list_models()` as configured in `app/core/llm.py`.

See [README.md](./README.md) for runtime configuration.
