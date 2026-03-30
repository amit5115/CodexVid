# Learn UI — Complete Click Guide

This document describes every UI element in the default web interface served at **`/`** and **`/learn`**: the files `learn.html`, `learn.js`, and `learn.css`.

**Default dev URL:** `http://127.0.0.1:8501/` (change with `VCAI_PORT`)

There is **no** legacy workspace SPA in this build — only the learn experience and the CodexVid API.

---

## Visual Structure

The UI has three screens that show one at a time:

```
┌─────────────────────────────────────────────────────┐
│  Header: "CodexVid AI"  |  Model dropdown            │
├─────────────────────────────────────────────────────┤
│  SCREEN 1: Upload                                    │
│  ┌───────────────────────────────────────────────┐  │
│  │  YouTube link input  /  File drop zone        │  │
│  │  [Process video]                              │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  SCREEN 2: Processing (spinner / status dots)        │
│                                                      │
│  SCREEN 3: Workspace  (65 / 35 split)                │
│  ┌──────────────────────┬────────────────────────┐  │
│  │  <video 16:9>        │ [Lesson]  [Chat]        │  │
│  │  sticky, left panel  ├────────────────────────┤  │
│  │                      │ Lesson tab:             │  │
│  │                      │  Chapter cards          │  │
│  │                      │  ├─ [mm:ss–mm:ss] pill  │  │
│  │                      │  └─ [💬 Ask about this] │  │
│  │                      │  Key takeaways / Quiz   │  │
│  │                      ├────────────────────────┤  │
│  │                      │ Chat tab:               │  │
│  │                      │  [segment context banner│  │
│  │                      │  + × clear button]      │  │
│  │                      │  Message log            │  │
│  │                      │  [text input] [send]    │  │
│  └──────────────────────┴────────────────────────┘  │
└─────────────────────────────────────────────────────┘
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

Shown after a successful upload. Layout: **65 / 35 grid** — video on the left, lesson/chat panel on the right.

### Video Panel

| Element | ID | What It Does |
|---------|-----|--------------|
| Video player | `#learnVideo` | HTML5 `<video controls>` — 16:9 aspect ratio, sticky at top |

**Source:** Set to `/api/codexvid/sessions/{session_id}/video` after upload completes.
The server streams `source.mp4`. Browser uses HTTP range requests for seeking.

**Auto-seek from chat:** When an assistant response includes a jump button (`📍`), clicking it calls `video.currentTime = timestamp_start` + `video.play()`. In segment mode, the timestamp always corresponds to the selected chapter.

**Active chapter tracking:** As the video plays, `timeupdate` fires ~4× per second. The current chapter (whose `start ≤ currentTime ≤ end`) gets an accent-border glow; the lesson panel auto-scrolls to keep it in view.

### Tab Bar

| Button | `data-tab` | What It Shows |
|--------|-----------|---------------|
| **Lesson** | `learn` | Teaching pack: chapters, key takeaways, quiz |
| **Chat** | `chat` | Segment context banner + message log + input |

Clicking **Lesson** also clears any active segment context.

---

## 6. Lesson Tab

Populated by `renderTeaching(teaching)` in `learn.js` using `payload.teaching` from the upload response. Chapter data is stored in `chaptersData[]` for active-tracking and segment chat.

### Chapter Cards

Each chapter renders as a card with:

| Element | Class | What It Does |
|---------|-------|--------------|
| Chapter title | `h3` inside `.chapter` | Static label from LLM |
| **Timestamp pill** | `.chapter-time-btn` | Teal clickable button showing `mm:ss – mm:ss`; click → seeks video to `start_time` + plays |
| **"💬 Ask about this"** | `.btn-segment-chat` | Opens Chat tab and auto-generates an explanation; no typing required |
| Description | `p` | LLM explanation for the segment |
| Active highlight | `.chapter.active` | Accent border + background glow; applied while video plays through this chapter |

**Chapter timestamp format:** All times shown as `m:ss` (e.g. `7:03 – 7:16`), never raw seconds.

**Active chapter:** `.chapter.active` is applied via `timeupdate` listener. When the playing position enters a chapter's time range, the card highlights and the lesson panel scrolls to it automatically.

### "💬 Ask about this" — Full Flow

1. Click the button on any chapter card
2. Chat tab opens; context banner appears: `Chatting about: "Chapter Title" (0:57 – 1:48)`
3. Video seeks to chapter start + plays
4. User message `"Explain this segment in simple terms"` appears in chat log
5. Animated **"Thinking…"** bubble appears immediately (3 bouncing dots)
6. `POST /api/codexvid/chat` fires with `segment_start`/`segment_end` — backend restricts FAISS hits to that time window
7. Thinking bubble is replaced by the AI explanation
8. Jump button in the response always points to the selected chapter's times
9. **Re-clicking the same chapter** → instant response from `segmentCache` (no API call)
10. All "Ask about this" buttons are disabled while a fetch is in-flight (debounce)

**Edge cases:**
- Empty transcript for segment → shows "No transcript available" without API call
- API error → thinking bubble replaced with friendly error; buttons re-enabled
- Segment cache cleared on new video upload

### Key Takeaways Section

Bulleted list from `teaching.key_takeaways`. Generated server-side from topic summaries.

### Quiz Section

Q&A pairs from `teaching.quiz`. Question + answer shown inline.

---

## 7. Chat Tab

### Segment Context Banner

| Element | ID | What It Does |
|---------|-----|--------------|
| Banner | `#segment-context` | Shown when a chapter is active; hidden for full-video chat |
| Label | `#segment-context-text` | `Chatting about: "Title" (mm:ss – mm:ss)` |
| Clear button `×` | `#segment-context-clear` | Clears segment context; returns to full-video chat mode |

When segment context is active:
- The chat input placeholder changes to `Ask a follow-up about "Chapter Title"…`
- Every query sent from the form is prefixed with the segment's transcript so the LLM scopes its answer
- `segment_start`/`segment_end` are sent to the API to filter FAISS hits
- Jump buttons always show the selected chapter's timestamps (not FAISS-derived ones)

### Chat Log

| Element | ID | What It Shows |
|---------|-----|---------------|
| Log container | `#chat-log` | User messages (right-aligned) + assistant messages |

**Assistant message format:**
- `📍 Jump to segment (mm:ss – mm:ss)` button — clicking seeks video
- Answer text (markdown `**bold**` → `<strong>`)
- Key points bulleted list
- Teaching mode badge (`[simple]`, `[detailed]`, etc.)

### Chat Input

| Element | ID | What It Does |
|---------|-----|--------------|
| Text area | `#chat-input` | Type a follow-up question; Enter submits |
| Submit button | `#chat-form button[type=submit]` | Sends query to `/api/codexvid/chat` |

**What happens when you submit:**
1. `chatBusy` guard checked — blocks concurrent submissions
2. `segmentContext` captured at submit-time (avoids stale closure if user switches chapter)
3. User message appended; textarea cleared
4. Query built: if segment active, full context prefix prepended
5. `POST /api/codexvid/chat` with `segment_start`/`segment_end` if segment active
6. Response rendered; jump button timestamp overridden with selected segment's times if active
7. `chatBusy` cleared

**Chat modes (auto-detected from query text):**

| Keyword pattern | Mode |
|----------------|------|
| "simple", "easy", "basic" | `simple` |
| "detail", "explain", "in depth" | `detailed` |
| "analogy", "like", "compare" | `analogy` |
| "example", "show me" | `example` |
| (default) | `default` |

---

## 8. APIs Called by the UI

| When | Method | Endpoint | Payload |
|------|--------|----------|---------|
| Process video clicked | `POST` | `/api/codexvid/upload` | FormData: file/youtube_url, model, whisper_model, language |
| After upload succeeds | `GET` | `/api/codexvid/sessions/{id}/video` | — (set as `<video src>`) |
| "Ask about this" click | `POST` | `/api/codexvid/chat` | JSON: session_id, query, model, segment_start, segment_end |
| Manual chat message | `POST` | `/api/codexvid/chat` | JSON: session_id, query, model, segment_start*, segment_end* |

\* `segment_start`/`segment_end` only included when a segment is active.

**Chat response fields used by UI:**

| Field | Used For |
|-------|---------|
| `answer` | Main text displayed in chat bubble |
| `timestamp_start` | Overridden by segment start when segment active; used for jump button |
| `timestamp_end` | Overridden by segment end when segment active |
| `key_points` | Bulleted list shown below answer |
| `chunks_used` | Used to decide whether to show jump button |
| `mode` | Teaching mode badge |

---

## 9. Seek Behavior Details

All video seeks use exact float seconds:

```javascript
// Chapter timestamp pill click
video.currentTime = ch.start;   // e.g. 57.0
video.play();

// Jump button in chat (📍 mm:ss – mm:ss)
video.currentTime = meta.timestamp_start;  // always the selected segment in segment mode
video.play();
```

**Timestamp display format:** `formatChapterTime(sec)` → `Math.floor(sec / 60) + ":" + padded_secs`. Example: `423.44 → "7:03"`. Never shows raw seconds in the UI.

---

## 10. Related Documentation

- [APP_DATAFLOW.md](./APP_DATAFLOW.md) — what happens on the server during upload and chat
- [ARCHITECTURE.md](./ARCHITECTURE.md) — component boundaries and module roles
- [TESTING.md](./TESTING.md) — automated test suite
- [README.md](./README.md) — quick start, configuration, API reference
