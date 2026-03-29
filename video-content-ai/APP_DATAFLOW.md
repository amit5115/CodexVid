# App data flow

This document describes how data moves through **CodexVid AI** for sessions: upload → transcribe → sentence timeline → semantic chunk → embed (FAISS) → sentence-refined chat + teaching pack.

## 1. Upload

1. Client sends `POST /api/codexvid/upload` with multipart field `file` (video) or a YouTube URL.
2. API saves or downloads the file and returns `session_id`, counts, and the teaching pack.
3. Processing runs in a thread-pool executor during the request: extract audio, run STT, write `transcript.json` (`segments` + `sentences`), semantic chunk, index, generate teaching JSON (sentence-based digest + snap when sentences exist).

## 2. Transcription (STT)

- **Local (`VCAI_STT_PROVIDER=whisper` or unset):** `faster-whisper` with **word-level timestamps** where available; overlapping audio windows use `VCAI_CODEXVID_CHUNK_SEC`, `VCAI_CODEXVID_AUDIO_OVERLAP_SEC`, and `VCAI_CODEXVID_PARALLEL_WORKERS`. Fine segments (`VCAI_CODEXVID_FINE_SEG_*`) group words into short spans with per-word `start`/`end`.
- **AWS (`VCAI_STT_PROVIDER=aws`):** audio is uploaded to S3, **AWS Transcribe** runs a batch job, result is polled and parsed into segments (normalized to the same chunking-friendly shape where possible).

Output segments include **`text`**, **`start`**, **`end`**, and usually **`words`** (each word: `word`, `start`, `end`). Timestamp alignment uses **no lead-in pad** so times match **`HTMLMediaElement.currentTime`** for seeking.

## 3. Sentence timeline

**`app/codexvid/timestamp_utils.py`** flattens words from all segments, groups them into **sentence spans** (punctuation-based), and can emit a full **`transcript_sentence_timeline`**.

**`transcript.json`** on disk is an object:

```json
{
  "segments": [ { "text", "start", "end", "words": [ ... ] } ],
  "sentences": [ { "text", "start", "end" } ]
}
```

## 4. Semantic chunking

**`app/codexvid/chunking.py`** no longer uses fixed word windows with overlap. It:

1. Builds sentences from word timings.
2. Packs sentences into chunks of roughly **`VCAI_CODEXVID_SEM_CHUNK_MIN_SEC`–`VCAI_CODEXVID_SEM_CHUNK_MAX_SEC`** (default **30–60** seconds), respecting sentence boundaries and splitting overly long spans at word timings.

Each chunk has **`text`**, **`start_time`**, **`end_time`** (float seconds), and **`start`/`end`** aliases for compatibility.

## 5. Embeddings + FAISS

Each chunk’s **text** is embedded; metadata stores **`start_time`**, **`end_time`**, plus **`start`/`end`** mirrors. Vectors live per session under `data/codexvid_sessions/<session_id>/` (see `app/codexvid/vector_store.py`).

## 6. Chat (RAG + sentence timestamps + multi-stage LLM)

1. Client sends `POST /api/codexvid/chat` with `session_id`, `query`, and optional `model` / `mode`. The handler passes **`session_id`** into the chat pipeline so **`transcript.json`** can be loaded from the session directory.

2. **Layer 1 — chunk retrieval (unchanged):** FAISS search returns the **top‑k** chunks only (`VCAI_CODEXVID_RAG_TOP_K`, default **3**). The full transcript is **not** sent to the LLM.

3. **Layer 2 — sentence pick (`app/codexvid/retrieval_utils.py`):**
   - Load **`sentences`** from `transcript.json` (or rebuild from `segments` / legacy list via `transcript_sentence_timeline`).
   - Keep sentences whose time range **overlaps** any retrieved chunk.
   - **One batched embedding call:** `[query] + [each candidate sentence text]` using `VCAI_EMBEDDING_MODEL` (same embedding path as FAISS indexing).
   - **Cosine similarity** picks the **single best** sentence; **`timestamp_start`** / **`timestamp_end`** in the API become that sentence’s **`start`** / **`end`**. If no sentences or embedding fails, timestamps fall back to the **chunk union**.

4. **Stage 1 — extraction:** Prompt asks for **all** relevant points from the retrieved chunk text **without** stripping technical detail (not a summary).

5. **Stage 2 — explanation:** Strict teacher system prompt; model returns **JSON** with `answer`, `timestamp_start`, `timestamp_end`, `key_points` using the **refined** sentence times (or chunk fallback). The prompt labels whether the span is **sentence-level** or **chunk-level**.

6. **Validation:** A **grounding score** checks that substantive answer tokens appear in the transcript excerpt; low confidence yields a safe fallback string (`Not clearly explained in this segment`).

**Example JSON fields in the HTTP response** (see OpenAPI): `answer`, `timestamps`, `timestamp_start`, `timestamp_end`, `key_points`, `grounded`, `grounding_score`, `mode`, `chunks_used`.

## 7. Teaching pack (post-upload)

- **`generate_teaching_output`** runs **one LLM call per semantic chunk** (no full-transcript chapter pass). Each call returns JSON: `topic_title`, `description`, and chunk **`start_time`/`end_time`** (enforced from the chunk for exact alignment).
- Results are **aggregated** in time order, **merge_adjacent_topics** merges only **consecutive** segments whose titles are highly similar (`difflib` ratio ≥ 0.90; stricter pass or no-merge if a long video would drop below ~5 topics).
- **`enforce_coverage`** extends the first/last topic to the full chunk timeline so timestamps span the video.
- Optional **`sentences`**: **`snap_chapter_times_to_sentences`** snaps chapter **`start`/`end`** to sentence boundaries for the Lesson tab.
- A **second** LLM call (summaries only) produces **`key_takeaways`** and **`quiz`** from the topic list—not the raw transcript.
- Response includes **`topics`** (structured list) and **`chapters`** (same content shaped for the existing UI).

## 8. Status and cleanup

- **Exists:** `GET /api/codexvid/sessions/{id}/exists` checks for an index file.
- **Video:** `GET /api/codexvid/sessions/{id}/video` streams the stored source copy.

## 9. Health

- **`GET /health`** — process up.
- **`GET /ready`** — checks LLM availability (`list_models` on the configured provider); does not require a database.

---

For UI steps, see [UI_CLICK_GUIDE.md](./UI_CLICK_GUIDE.md). For code structure, see [ARCHITECTURE.md](./ARCHITECTURE.md).
