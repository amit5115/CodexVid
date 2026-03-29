# Learn UI — Complete Click Guide

This document describes every UI element in the default web interface served at **`/`** and **`/learn`**: the files `learn.html`, `learn.js`, and `learn.css`.

**Default dev URL:** `http://127.0.0.1:8501/` (change with `VCAI_PORT`)

There is **no** legacy workspace SPA in this build — only the learn experience and the CodexVid API.

---

## Visual Structure

The UI has three screens that show one at a time:

```
┌─────────────────────────────────────────────────┐
│  Header: "CodexVid AI"  |  Model dropdown        │
├─────────────────────────────────────────────────┤
│                                                  │
│  SCREEN 1: Upload                                │
│  ┌──────────────────────────────────────────┐   │
│  │  YouTube link input                      │   │
│  │  File drop zone                          │   │
│  │  [Process video]                         │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  SCREEN 2: Processing (spinner / status)         │
│                                                  │
│  SCREEN 3: Workspace                             │
│  ┌──────────────┬─────────────────────────────┐ │
│  │              │ [Lesson] [Chat]              │ │
│  │  <video>     ├─────────────────────────────┤ │
│  │              │ Tab content:                 │ │
│  │              │  Lesson: chapters + quiz     │ │
│  │              │  Chat:   message log + input │ │
│  └──────────────┴─────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 1. Page Load (`/` or `/learn`)

**What happens:**
- Server (`app/main.py`) serves `learn.html` with `Cache-Control: no-cache` headers
- Browser loads `learn.css` and `learn.js` from `/static/`
- Screen 1 (Upload) is shown; Screens 2 and 3 are hidden (`display: none`)
- Background animated orbs start (CSS keyframe animation)

**What you see:**
- Header with brand name and tagline
- Model dropdown (top right)
- Upload card with YouTube URL input, file drop zone, and "Process video" button

---

## 2. Header

| Element | ID / Class | What It Does |
|---------|-----------|--------------|
| Brand name | `.brand` | Static display text — "CodexVid AI" |
| Tagline | `.tagline` | Static display text |
| **Model dropdown** | `#learn-model` | Selects LLM model. Value is sent as `model` on every upload and chat request. Options: `llama3`, `llama3.1`, `mistral`, `gpt-4o-mini`, `gpt-4o`, `gpt-4.1`, `claude-sonnet-4-20250514`, etc. |

**Backend impact of Model selection:**
- Upload: `model` field in form data → used for teaching pack LLM calls
- Chat: `model` field in JSON body → used for extraction + explanation LLM calls
- The provider is selected by `app/core/llm.py` `get_provider(model)` — company GPT models auto-route to CompanyGPTProvider

---

## 3. Screen 1: Upload

### YouTube Link Input

| Element | ID | What It Does |
|---------|-----|--------------|
| URL text field | `#youtube-url` | Enter a YouTube video URL |

**Behavior:**
- If non-empty after `.trim()`, the YouTube URL takes priority over the file input
- The value is sent as `youtube_url` in the multipart form data
- The server validates it is a real YouTube URL (rejects other domains with HTTP 422)
- Backend: `app/services/video.py` → `download_video(url, tmpdir)` via yt-dlp
- yt-dlp uses socket timeout (`VCAI_YTDLP_SOCKET_TIMEOUT`, default 120s) and iOS/Android client fallback to avoid JS-challenged streams

### File Drop Zone

| Element | ID | What It Does |
|---------|-----|--------------|
| Drop zone container | `#dropzone` | Visual drag-and-drop target |
| Hidden file input | `#file-video` | Actual `<input type="file">` |

**Behavior:**
- Click the drop zone → opens file picker
- Drag a video file over → highlight; drop → file selected
- Only used if the YouTube URL field is empty
- Accepted formats: any video file (MP4, MOV, MKV, etc.); FFmpeg handles extraction on the backend

### Process Video Button

| Element | ID | What It Does |
|---------|-----|--------------|
| Submit button | `#btn-start-upload` | Triggers upload and pipeline |

**What happens when clicked:**
1. Validates: YouTube URL or file must be provided; if neither → shows inline error, no request sent
2. Switches UI to Screen 2 (Processing)
3. Builds `FormData`:
   ```
   youtube_url  (if YouTube field non-empty)
   file         (if file selected and no URL)
   model        (from #learn-model)
   whisper_model: "base"
   language:    "en"
   ```
4. Sends `POST /api/codexvid/upload` with the form data
5. While waiting: shows the Processing screen with a spinner

---

## 4. Screen 2: Processing

**Shown while:** transcription, chunking, FAISS embedding, and teaching pack generation run on the server.

| Element | What It Shows |
|---------|--------------|
| Status card | "Building your lesson..." message |
| Spinner animation | CSS animation indicating background work |

**What's happening on the server during this screen:**
1. FFmpeg extracts 16 kHz mono WAV from the video
2. faster-whisper (or AWS Transcribe) transcribes with word-level timestamps
3. Word → sentence grouping
4. Semantic chunking (30–60s windows, sentence-boundary-aware)
5. FAISS index built from chunk embeddings
6. Parallel LLM calls (one per chunk) generate topics/descriptions
7. Topic merging, coverage enforcement, optional sentence snap
8. Second LLM call generates key takeaways and quiz
9. All artifacts saved to `data/codexvid_sessions/{session_id}/`
10. Server responds with JSON

**When upload finishes:** `learn.js` receives the JSON response, renders the teaching content, sets the video source, and switches to Screen 3.

---

## 5. Screen 3: Workspace

Shown after a successful upload. Contains three areas: video panel, tab bar, and tab content.

### Video Panel

| Element | ID | What It Does |
|---------|-----|--------------|
| Video player | `#learnVideo` | HTML5 `<video controls>` element |

**Source:** Set to `/api/codexvid/sessions/{session_id}/video` after upload completes.
The server streams `source.mp4` from the session directory. Browser uses HTTP range requests for seeking.

**Seeking from chat:** `learn.js` calls `video.currentTime = timestamp_start` (exact float seconds, no rounding) when a chat response includes a timestamp. This jumps to the sentence the LLM identified as most relevant to your question.

### Tab Bar

| Button | `data-tab` | What It Shows |
|--------|-----------|---------------|
| **Lesson** | `learn` | Teaching pack: chapters, key takeaways, quiz |
| **Chat** | `chat` | Chat log + input form |

Clicking a tab shows its panel and hides the other. Active tab gets a highlighted style.

---

## 6. Lesson Tab

Populated by `renderTeaching(teaching)` in `learn.js` using `payload.teaching` from the upload response.

### Chapters / Topics Section

Each topic from `teaching.topics` (or `teaching.chapters`) renders as:

| Element | Content |
|---------|---------|
| Chapter title | `topic_title` from LLM |
| Time range | Formatted as `mm:ss – mm:ss` from `start_time`/`end_time` |
| Description | `description` from LLM |
| Clickable timestamp | Clicking the time range → sets `video.currentTime` to `start_time` |

Topics are sorted chronologically and cover the full video duration (enforced by `enforce_coverage()` on the server).

### Key Takeaways Section

A bulleted list of learning points from `teaching.key_takeaways`. Generated by a second LLM call from the topic summaries (not the raw transcript).

### Quiz Section

Q&A pairs from `teaching.quiz`. Each item shows:
- Question text
- Toggle/reveal answer (or shown inline depending on CSS)

---

## 7. Chat Tab

### Chat Log

| Element | ID | What It Shows |
|---------|-----|---------------|
| Log container | `#chat-log` | Scrollable list of user and assistant messages |

**User message format:** Plain text, right-aligned or styled differently from assistant.

**Assistant message format:**
- Answer text with basic markdown rendering (`**bold**` → `<strong>`, etc.)
- Key points list (from `result.key_points`)
- Timestamp button (e.g. `📍 00:12`) — clicking seeks video to that timestamp
- Mode indicator (e.g. `[simple]`, `[detailed]`)

### Chat Input

| Element | ID | What It Does |
|---------|-----|--------------|
| Text area | `#chat-input` | Type your question about the video |
| Submit button | part of `#chat-form` | Send the question |

**Keyboard shortcut:** Enter submits the form (Shift+Enter for newline, depending on JS config).

**What happens when you send a message:**
1. `chatBusy` flag is set (prevents concurrent submissions)
2. User message appended to chat log
3. Textarea cleared
4. Sends `POST /api/codexvid/chat`:
   ```json
   {
     "session_id": "abc123...",
     "query": "What is gradient descent?",
     "model": "llama3"
   }
   ```
5. Waits for response (no streaming in current build)
6. Appends assistant message to chat log
7. If `result.chunks_used > 0` and `result.timestamp_start` is defined: seeks video to `timestamp_start`
8. `chatBusy` cleared

**Chat modes (auto-detected from query keywords):**
- `simple` — plain explanation
- `detailed` — thorough technical breakdown
- `analogy` — explains using a real-world comparison
- `example` — provides a concrete example
- `default` — balanced teacher style

---

## 8. APIs Called by the UI

| When | Method | Endpoint | Payload |
|------|--------|----------|---------|
| Process video clicked | `POST` | `/api/codexvid/upload` | FormData: file/youtube_url, model, whisper_model, language |
| After upload succeeds | `GET` | `/api/codexvid/sessions/{id}/video` | — (set as `<video src>`) |
| Chat message sent | `POST` | `/api/codexvid/chat` | JSON: session_id, query, model |

**Chat response fields used by UI:**

| Field | Used For |
|-------|---------|
| `answer` | Main text displayed in chat |
| `timestamp_start` | `video.currentTime = timestamp_start` for seeking |
| `timestamp_end` | Displayed as end of timestamp range |
| `key_points` | Bulleted list shown below answer |
| `chunks_used` | Gate for whether to seek (seek only if > 0) |
| `grounded` | (not displayed but available) |
| `grounding_score` | (not displayed but available) |
| `mode` | Mode badge shown next to answer |
| `timestamps` | Legacy `📍 mm:ss` seek buttons in answer text |

---

## 9. Seek Behavior Details

The UI uses **exact float seconds** for all video seeks:

```javascript
// Primary seek (from chat API response)
video.currentTime = result.timestamp_start;  // e.g. 12.345

// Secondary seek (from 📍 labels in answer text — legacy)
const parts = label.split(":");
const seconds = parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
video.currentTime = seconds;
```

The `timestamp_start` value is the `start` time of the **best-matching sentence** within the retrieved FAISS chunks (determined by cosine similarity between the query embedding and sentence embeddings). This gives sub-chunk precision — usually pointing to the exact spoken sentence most relevant to your question.

---

## 10. Related Documentation

- [APP_DATAFLOW.md](./APP_DATAFLOW.md) — what happens on the server during upload and chat
- [ARCHITECTURE.md](./ARCHITECTURE.md) — component boundaries and module roles
- [TESTING.md](./TESTING.md) — automated test suite
- [README.md](./README.md) — quick start, configuration, API reference
