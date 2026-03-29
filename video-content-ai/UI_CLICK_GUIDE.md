# Learn UI — what each control does

This document describes the **default** web interface served at **`/`** and **`/learn`**: **`learn.html`**, **`learn.js`**, **`learn.css`**. Default dev URL: **`http://127.0.0.1:8501/`** (set **`VCAI_PORT`** to change).

There is **no** legacy workspace SPA in this build—only the learn experience and the CodexVid API.

---

## 1. First load (`/` or `/learn`)

- **Server:** `app/main.py` serves the same HTML for both routes (`learn.html`).
- **Assets:** `learn.css`, `learn.js` from `/static/`.
- **You see:** header **CodexVid** (or product title from HTML), tagline about upload → learn → ask, **Model** dropdown, **Learn from a video** with **YouTube link** and/or **file upload**, **Process video**.

---

## 2. Header

| Element | What it does |
|--------|----------------|
| Title / subtitle | Display only. |
| **Model** | Sent as `model` on `POST /api/codexvid/upload` and `POST /api/codexvid/chat`. |

---

## 3. Upload card

| Element | What it does |
|--------|----------------|
| **YouTube link** | If non-empty after trim, submit uses **YouTube mode** (`youtube_url` in form data, no file). |
| **File** | Used when the URL field is empty; `file` is appended to the multipart form. |
| **Process video** | Requires URL **or** file; then `POST /api/codexvid/upload` with `model`, `whisper_model`, `language`, etc. |

**Backend:** `app/api/codexvid.py` — YouTube URLs are validated and downloaded via `app/services/video.py`; local files go through the same `process_upload` pipeline in `app/codexvid/session.py` (transcript with `segments` + `sentences`, semantic chunks, FAISS, teaching pack built with **sentence** lines and **snapped** chapter times when sentences exist).

---

## 4. Processing screen

Shown while transcription, chunking, embedding (FAISS), and teaching-pack generation run.

- **Local STT:** **faster-whisper** after ffmpeg extracts audio (see `VCAI_WHISPER_*` and CodexVid slice settings in `app/config.py`).
- **AWS:** With **`VCAI_STT_PROVIDER=aws`**, audio is sent through **Amazon Transcribe** instead of local Whisper.

---

## 5. After success (workspace view)

| Area | What it does |
|------|----------------|
| **Video** | Session copy served from **`GET /api/codexvid/sessions/{session_id}/video`**. |
| **Lesson** tab | Renders structured teaching content from the upload response. |
| **Chat** tab | User messages and assistant replies; **jump** control uses **`timestamp_start`** from the chat API (float seconds) when **`chunks_used` > 0**. The backend sets that time to the **best-matching sentence** inside retrieved chunks (embedding similarity), not the whole 30–60s chunk, when `transcript.json` has sentences. Optional **Key points** from `key_points`. Legacy lines starting with **📍** still seek using **fractional** `mm:ss` parsing (`parseFloat` on seconds). |
| **Send** | **`POST /api/codexvid/chat`** with `session_id`, `query`, optional `model` / `mode`. |

**Seek behavior:** `learn.js` sets **`video.currentTime = timestamp_start`** using the **exact** float from the server (no integer rounding). That value is usually the **start of the most relevant spoken sentence** for the answer, falling back to chunk bounds if needed.

---

## 6. APIs used by this UI

| Endpoint | When |
|----------|------|
| `POST /api/codexvid/upload` | After **Process video** |
| `GET /api/codexvid/sessions/{id}/video` | Sets `<video src>` |
| `POST /api/codexvid/chat` | Chat submit |

**Chat response shape** (subset): `answer`, `timestamp_start`, `timestamp_end`, `key_points`, `chunks_used`, `grounded`, `grounding_score`, `mode`, `timestamps`.

Open **`/docs`** for the full OpenAPI list (including exists and other codexvid routes).

---

## 7. Related docs

- [APP_DATAFLOW.md](./APP_DATAFLOW.md) — pipeline steps (semantic chunking, multi-stage LLM).
- [ARCHITECTURE.md](./ARCHITECTURE.md) — components.
- [TESTING.md](./TESTING.md) — automated tests.
