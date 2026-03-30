# App Data Flow (CodexVid AI)

This document describes exactly how data moves through the application for every major operation. Start here to understand what happens step by step when a user uploads a video, asks a question, or receives the teaching pack.

---

## 1. Upload Flow

### 1.1 Client Request

The browser (`learn.js`) sends:

```
POST /api/codexvid/upload
Content-Type: multipart/form-data

file: <binary video data>         (or omit and send youtube_url)
youtube_url: "https://..."        (or omit and send file)
model: "llama3"
whisper_model: "base"
language: "en"
```

### 1.2 FastAPI Handler (`app/api/codexvid.py`)

1. Validates: exactly one of `file` or `youtube_url` must be present; `youtube_url` must be a real YouTube URL if provided
2. If YouTube URL: calls `download_video(url, tmpdir)` via `app/services/video.py` (yt-dlp) → local MP4 path
3. If file upload: writes bytes to a temp file
4. Calls `process_upload(video_path, whisper_model, language, model)` in a `ThreadPoolExecutor` (keeps async FastAPI non-blocking)
5. Returns `session_id`, `segment_count`, `chunk_count`, `teaching`, `source`, `youtube_url`

### 1.3 `process_upload()` (`app/codexvid/session.py`)

Full pipeline executed synchronously in the thread pool:

```
video_path (MP4 or other)
    │
    ├─[1] new_session_dir() → UUID session_id + disk path
    │
    ├─[2] Copy video → {session_dir}/source.mp4
    │
    ├─[3] transcribe_video(source.mp4, ...) → segments
    │         └── extract_audio_wav() via FFmpeg (16 kHz mono WAV)
    │         └── _split_audio() → overlapping windows
    │         └── parallel _transcribe_one_chunk() × N workers
    │         └── merge_segments() + deduplicate_overlapping_words()
    │         └── returns: list[{text, start, end, words:[{word,start,end}]}]
    │
    ├─[4] transcript_sentence_timeline(segments) → sentence list
    │
    ├─[5] Save transcript.json → {segments: [...], sentences: [...]}
    │
    ├─[6] create_chunks(segments) → semantic_chunks
    │         └── flatten_words_from_transcript()
    │         └── words_to_sentence_spans()
    │         └── greedy pack into 30–60s windows
    │         └── split oversized sentences at word boundaries
    │         └── returns: list[{text, start_time, end_time, start, end}]
    │
    ├─[7] Save chunks.json
    │
    ├─[8] CodexvidVectorStore.build(chunks, session_dir, embed_fn)
    │         └── batch embed all chunk texts (VCAI_EMBEDDING_MODEL)
    │         └── L2-normalize vectors
    │         └── faiss.IndexFlatIP.add(normalized_vectors)
    │         └── Save faiss.index + faiss_meta.json
    │
    ├─[9] generate_teaching_output(chunks, sentences, model, workers)
    │         └── parallel LLM call per chunk (VCAI_CODEXVID_TEACHING_CHUNK_WORKERS)
    │         └── merge_adjacent_topics() (title similarity ≥ 0.90)
    │         └── enforce_coverage() (extend first/last to video bounds)
    │         └── snap_chapter_times_to_sentences() (if sentences available)
    │         └── second LLM call: takeaways + quiz from topic summaries
    │
    └─[10] Save teaching.json
           Return: (session_id, {segment_count, chunk_count, teaching})
```

### 1.4 Client Response Handling

`learn.js` receives the JSON response and:
1. Stores `session_id` in global `sessionId`
2. Calls `renderTeaching(payload.teaching)` → populates the Lesson tab
3. Sets `<video src>` to `/api/codexvid/sessions/{sessionId}/video`
4. Switches from the Processing screen to the Workspace screen

---

## 2. Transcription Detail

### 2.1 Local Whisper (`VCAI_STT_PROVIDER=whisper` or unset)

```
source.mp4
    │
    ▼ FFmpeg
16kHz mono WAV
    │
    ▼ _split_audio(chunk_duration=25s, overlap=5s)
[window_0: 0–25s], [window_1: 20–45s], [window_2: 40–65s], ...
                    ↑ step = chunk_duration - overlap = 20s
    │
    ▼ parallel faster-whisper (N workers)
per-window segments with word-level timestamps
    │
    ▼ merge_segments() + deduplicate_overlapping_words()
unified segment list, no duplicate words in overlap regions
    │
    ▼ return
list[{text, start, end, words:[{word, start, end}]}]
```

**Why overlapping windows?** Whisper sometimes drops the first/last words in a clip. The 5-second overlap ensures boundary words are captured by at least one window.

**Word deduplication:** Words from overlapping regions are de-duplicated by comparing `start` timestamps. The word with the more accurate (earlier window's) timestamp is kept.

### 2.2 AWS Transcribe (`VCAI_STT_PROVIDER=aws`)

```
source.mp4 audio track
    │
    ▼ boto3 S3 upload → s3://{BUCKET}/{key}.wav
    │
    ▼ transcribe_client.start_transcription_job(...)
Job ID returned
    │
    ▼ poll every N seconds until COMPLETED or FAILED (timeout: 3600s)
    │
    ▼ download Transcribe JSON from S3
    │
    ▼ parse_transcript_json_to_segments()
Normalize to Whisper-compatible format: {text, start, end, words:[...]}
```

AWS language mapping: short codes like `en` → `en-US`, `es` → `es-US`. `auto` → `IdentifyLanguage=True`.

---

## 3. Sentence Timeline Construction

### Module: `app/codexvid/timestamp_utils.py`

```
segments: [{text, start, end, words:[{word, start, end}]}]
    │
    ▼ flatten_words_from_transcript()
flat_words: [{word, start, end}, ...]
    (if words missing from segment: linear interpolate timestamps)
    │
    ▼ deduplicate_overlapping_words()
    (remove words with duplicate start times from overlapping windows)
    │
    ▼ words_to_sentence_spans()
    (group words into sentences by punctuation: . ? ! ; \n)
    (force sentence break every 45 s even without punctuation — prevents mega-sentence
     when Whisper omits punctuation, e.g. with the `base` model on casual speech)
sentences: [{text, start, end, words:[...]}, ...]
    │
    ▼ transcript_sentence_timeline()
sentences without word-level detail: [{text, start, end}, ...]
```

**Saved to `transcript.json`:**
```json
{
  "segments": [... all segments with words ...],
  "sentences": [... sentences without word lists ...]
}
```

---

## 4. Semantic Chunking

### Module: `app/codexvid/chunking.py`

```
segments (with word timestamps)
    │
    ▼ Build sentences (via timestamp_utils)
[s1: 0–3.2s], [s2: 3.5–7.1s], [s3: 7.4–12.0s], ...
    │
    ▼ Greedy sentence packing:

    chunk_start = s1.start
    current_duration = 0

    while sentences remain:
        add next sentence → current_duration increases
        if current_duration >= SEM_CHUNK_MAX_SEC:
            → flush current chunk, start new
        elif current_duration >= SEM_CHUNK_MIN_SEC:
            → check: would next sentence push over max?
            → if yes: flush; if no: keep adding
    │
    ▼ Handle oversized single sentences:
    (split at word boundaries when one sentence > SEM_CHUNK_MAX_SEC)
    (if sentence has no word timestamps, produce one covering chunk — no silent drop)
    │
    ▼ Post-validation: minimum chunk count enforcement
    expected_min = floor(video_duration / SEM_CHUNK_MAX_SEC)
    if len(word_based_chunks) < expected_min:
        → switch to _chunk_segments_by_time() time-based fallback
        (groups segments directly by elapsed time, no word timestamp dependency)
    │
    ▼ return
chunks: [{text, start_time, end_time, start, end}, ...]
```

Each chunk is roughly 30–60 seconds and always ends at a sentence boundary (never mid-sentence). The time-based fallback guarantees multiple chunks even when Whisper word timestamps are missing or sparse.

---

## 5. FAISS Index Construction

### Module: `app/codexvid/vector_store.py`

```
chunks: [{text, start_time, end_time}, ...]
    │
    ▼ batch embed all texts
    get_provider().embed(model=VCAI_EMBEDDING_MODEL, texts=[chunk.text for chunk in chunks])
    → np.ndarray shape (N, D)  where D = embedding dimension (e.g. 768)
    │
    ▼ L2-normalize each vector
    vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    → cosine similarity via inner product
    │
    ▼ faiss.IndexFlatIP(D)
    index.add(normalized_vectors)
    │
    ▼ save to disk:
    faiss.write_index(index, "{session_dir}/faiss.index")
    json.dump({"dim": D, "meta": [...]}, "{session_dir}/faiss_meta.json")
```

**Why `IndexFlatIP`?** Flat (brute-force) index gives exact nearest neighbors. After L2 normalization, inner product equals cosine similarity. For typical video lengths (< 100 chunks), exact search is fast enough.

---

## 6. Chat Flow

### 6.1 Client Request

```
POST /api/codexvid/chat
Content-Type: application/json

{
  "session_id": "abc123...",
  "query": "What is gradient descent?",
  "model": "llama3",
  "mode": "simple",
  "segment_start": 57.0,   // optional — present when user clicked "Ask about this"
  "segment_end": 108.0     // optional — restricts FAISS hits to this time window
}
```

### 6.2 FastAPI Handler

1. Calls `load_store(session_id)` → loads `faiss.index` + `faiss_meta.json`
2. Calls `store.search(query, k=RAG_TOP_K)` → top-k chunks with scores
3. **Segment filtering** (when `segment_start`/`segment_end` are present): keeps only hits whose `[start_time, end_time]` overlaps the requested window; falls back to all hits if none overlap
4. Calls `chat(query, filtered_chunks, model, mode, session_id)` in thread pool

### 6.3 `chat()` Pipeline (`app/codexvid/chat.py`)

```
query: "What is gradient descent?"
retrieved_chunks: [{text, start_time, end_time}, ...]   (top-k from FAISS)
session_id: "abc123..."
    │
    ├─[Layer 1] FAISS already done (passed in)
    │
    ├─[Layer 2] Sentence-level timestamp refinement:
    │    load_session_sentences(session_id)
    │    → read transcript.json → sentences list
    │    │
    │    filter_sentences_overlapping_chunks(sentences, chunks)
    │    → keep sentences where [start,end] overlaps any chunk's [start_time,end_time]
    │    │
    │    find_most_relevant_sentence(query, candidate_sentences, embed_model)
    │    → ONE batch embed call: [query] + [sentence.text for each candidate]
    │    → cosine_similarity_matrix(query_vec, sentence_matrix)
    │    → return sentence with highest score
    │    │
    │    timestamp_start = best_sentence.start
    │    timestamp_end   = best_sentence.end
    │    (fallback: union of all retrieved chunks if no sentences or embed fails)
    │
    ├─[Stage 1] Extraction:
    │    system: "You are a precise information extractor."
    │    user:   "Extract ALL important points from this transcript: {chunks_text}"
    │    → LLM returns bullet list (no summarization, no detail loss)
    │    │
    │    if extraction is empty or "not mentioned":
    │        return early: "This topic is not covered in the video."
    │
    ├─[Stage 2] Explanation:
    │    system: "You are a teacher. {mode-specific style instructions}"
    │    user:   "Explain: {extraction}\n\nTimestamp (sentence-level): {start}–{end}\n\nReturn JSON."
    │    → LLM returns:
    │       {
    │         "answer": "Detailed explanation...",
    │         "timestamp_start": 12.345,
    │         "timestamp_end": 45.678,
    │         "key_points": ["point 1", "point 2"]
    │       }
    │
    ├─[Validation] Grounding check:
    │    answer_tokens = [t for t in answer.split() if len(t) >= 4]
    │    transcript_tokens = set(chunks_text.lower().split())
    │    grounding_score = |answer_tokens ∩ transcript_tokens| / |answer_tokens|
    │    if grounding_score < 0.38 and len(answer_tokens) >= 6:
    │        answer = "Not clearly explained in this segment."
    │
    └─[Return]
       {
         answer, timestamp_start, timestamp_end,
         key_points, grounded, grounding_score,
         mode, chunks_used, timestamps: [{start_label, end_label, start_sec, end_sec}]
       }
```

### 6.4 Client Response Handling

```javascript
result = await fetch('/api/codexvid/chat', ...).json()

// When a segment is active (user clicked "Ask about this"), the frontend
// overrides the returned timestamp with the SELECTED segment's boundaries
// so the jump button always points to the correct chapter.
const activeSeg = segmentContext   // captured before the await
const meta = activeSeg
  ? { timestamp_start: activeSeg.start, timestamp_end: activeSeg.end,
      key_points: result.key_points, chunks_used: result.chunks_used || 1 }
  : { timestamp_start: result.timestamp_start, timestamp_end: result.timestamp_end,
      key_points: result.key_points, chunks_used: result.chunks_used }

appendAssistantMessage(result.answer, result.mode, meta)

// Jump button in the message seeks video.currentTime = meta.timestamp_start
// (always in mm:ss display, exact float for seek)
```

---

## 7. Teaching Pack Generation

### Module: `app/codexvid/teaching.py`

```
chunks: [{text, start_time, end_time}, ...]   (semantic chunks, 30–60s each)
sentences: [{text, start, end}, ...]           (from transcript.json)
model: "llama3"
    │
    ├─[Phase 1] Parallel per-chunk LLM calls (N=VCAI_CODEXVID_TEACHING_CHUNK_WORKERS)
    │    For each chunk:
    │        system: "You are a precise segment analyzer. Describe ONLY what is in this
    │                 excerpt. NEVER summarize the whole video. NEVER use phrases like
    │                 'in this video' or 'throughout the video'."
    │        user:   "Text: {chunk.text}  Time: {start_time}–{end_time}
    │                 Return JSON: {topic_title, description, start_time, end_time}"
    │        → LLM returns topic JSON
    │        → Post-check: if description contains whole-video phrases
    │          ("in this video", "throughout the video", "the video covers", etc.)
    │          → substitute raw transcript snippet as description
    │        (fallback: use first 100 chars as title, generic description, if LLM fails)
    │    → topic_list: [{topic_title, description, start_time, end_time}, ...]
    │
    ├─[Phase 2] Post-processing:
    │    merge_adjacent_topics(topic_list):
    │        → Compare consecutive topic titles with difflib.SequenceMatcher
    │        → Merge if ratio ≥ 0.90 AND total topics would stay ≥ 5 (for long videos)
    │
    │    enforce_coverage(topic_list, chunks):
    │        → Triggered for videos ≥ 120 s (LONG_VIDEO_SEC, down from 300 s)
    │        → topic_list[0].start_time  = chunks[0].start_time   (video start)
    │        → topic_list[-1].end_time   = chunks[-1].end_time     (video end)
    │
    │    snap_chapter_times_to_sentences(topic_list, sentences):
    │        → For each topic boundary, find nearest sentence start/end
    │        → Snap to sentence boundary for cleaner chapter alignment
    │
    ├─[Phase 3] Second LLM call (summaries only, not raw transcript):
    │    prompt: "You create study aids from segment summaries.
    │             Topics: {topic_summaries}
    │             Return JSON: {key_takeaways: [...], quiz: [{question, answer}]}"
    │    → key_takeaways: list of 3–7 learning points
    │    → quiz: list of 3–5 Q&A pairs
    │
    └─[Return]
       {
         "topics": [{topic_title, description, start_time, end_time}, ...],
         "chapters": [same content, shaped for UI rendering],
         "key_takeaways": ["...", ...],
         "quiz": [{"question": "...", "answer": "..."}, ...]
       }
```

---

## 8. Session Video Streaming

```
GET /api/codexvid/sessions/{session_id}/video
    │
    ▼ Resolve path: {CODEXVID_SESSIONS_DIR}/{session_id}/source.mp4
    │
    ▼ Read file in chunks (streaming response)
    │
    ▼ HTTP response:
    Content-Type: video/mp4
    (or auto-detected MIME type)
    Body: binary video stream
```

The `<video>` element in `learn.html` uses the browser's built-in range request support for seeking.

---

## 9. Health & Readiness

### `GET /health`
```
Always returns 200:
{ "status": "ok", "version": "1.0.0", "product": "codexvid-ai" }
```

### `GET /ready`
```
calls get_provider().list_models()
    │
    ├─ Success → 200: { "status": "ready", "checks": { "llm": "ok" } }
    └─ Exception → 503: { "status": "not_ready", "checks": { "llm": "error: ..." } }
```

---

## 10. Disk I/O Summary

| Operation | Files Written | Files Read |
|-----------|--------------|------------|
| Upload | `source.mp4`, `transcript.json`, `chunks.json`, `faiss.index`, `faiss_meta.json`, `teaching.json` | — |
| Chat | — | `faiss.index`, `faiss_meta.json`, `transcript.json` |
| Video stream | — | `source.mp4` |
| Exists check | — | `faiss.index` (stat only) |

---

## 11. Error Handling

| Scenario | Behavior |
|----------|----------|
| No file and no YouTube URL | HTTP 422 Unprocessable Entity |
| Invalid YouTube URL (not YouTube) | HTTP 422 with message |
| YouTube download fails | HTTP 500 with yt-dlp error |
| Whisper fails on a window | That window skipped; rest proceed |
| LLM call fails in teaching | Fallback topic with snippet title |
| LLM returns invalid JSON in chat | Stage 2 retried with stricter prompt; fallback answer used |
| Grounding score too low | Answer replaced with "Not clearly explained in this segment." |
| Session not found on chat | HTTP 404 |
| FAISS index missing on exists check | Returns `{"exists": false}` |

---

For component roles, see [ARCHITECTURE.md](./ARCHITECTURE.md).
For UI walkthrough, see [UI_CLICK_GUIDE.md](./UI_CLICK_GUIDE.md).
