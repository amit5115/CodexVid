"""CodexVid API: upload → teach pack; chat with timestamps (FAISS)."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import CODEXVID_RAG_TOP_K, CODEXVID_SESSIONS_DIR, DEFAULT_MODEL
from app.codexvid.chat import chat as codexvid_chat
from app.codexvid.chat import detect_mode
from app.codexvid.session import load_store, process_upload
from app.services.video import download_video, normalize_media_source

logger = logging.getLogger(__name__)


def _is_youtube_url(url: str) -> bool:
    u = url.lower()
    return "youtube.com" in u or "youtu.be" in u


router = APIRouter(prefix="/api/codexvid", tags=["codexvid"])


class CodexvidChatBody(BaseModel):
    session_id: str = Field(..., min_length=8)
    query: str = Field(..., min_length=1)
    model: str | None = None
    mode: str | None = None  # optional override; else auto-detect


@router.post("/upload")
async def codexvid_upload(
    file: UploadFile | None = File(None),
    youtube_url: str = Form(""),
    whisper_model: str = Form("base"),
    language: str = Form("en"),
    model: str = Form(DEFAULT_MODEL),
):
    """Upload a local video **or** paste a **YouTube** URL — transcribe, chunk, FAISS, teaching pack."""
    loop = asyncio.get_event_loop()
    yt = (youtube_url or "").strip()

    if yt:
        url = normalize_media_source(yt)
        if not _is_youtube_url(url):
            return JSONResponse(
                {"error": "URL mode supports YouTube only (youtube.com or youtu.be)."},
                status_code=400,
            )

        def _from_youtube() -> tuple[str, dict]:
            tmp_dir = Path(tempfile.mkdtemp(prefix="codexvid-yt-"))
            try:
                video_path = download_video(url, tmp_dir)
                return process_upload(
                    video_path,
                    whisper_model=whisper_model,
                    language=language,
                    llm_model=model,
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        try:
            _sid, payload = await loop.run_in_executor(None, _from_youtube)
            payload = {**payload, "source": "youtube", "youtube_url": url}
            return payload
        except FileNotFoundError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            logger.exception("CodexVid YouTube upload failed")
            return JSONResponse({"error": str(e)}, status_code=500)

    if file is None:
        return JSONResponse(
            {"error": "Upload a video file or provide a YouTube URL."},
            status_code=400,
        )

    suffix = Path(file.filename or "video").suffix or ".mp4"
    if suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}:
        return JSONResponse(
            {"error": "Unsupported video type; use mp4, webm, mov, mkv, avi, m4v"},
            status_code=400,
        )

    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        sid, payload = await loop.run_in_executor(
            None,
            lambda: process_upload(
                tmp_path,
                whisper_model=whisper_model,
                language=language,
                llm_model=model,
            ),
        )
        payload = {**payload, "source": "upload"}
        return payload
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception("CodexVid upload failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/chat")
async def codexvid_chat_endpoint(body: CodexvidChatBody):
    """RAG chat grounded in session chunks; strict out-of-context phrase."""
    try:
        store = load_store(body.session_id)
    except FileNotFoundError:
        return JSONResponse({"error": "Unknown session_id"}, status_code=404)

    loop = asyncio.get_event_loop()
    k = max(1, min(CODEXVID_RAG_TOP_K, store.index.ntotal or 1))
    hits = await loop.run_in_executor(
        None,
        lambda: store.search(body.query.strip(), k=k),
    )
    model = body.model or DEFAULT_MODEL
    explicit = body.mode.strip() if body.mode else ""
    mode = explicit if explicit else detect_mode(body.query)
    result = await loop.run_in_executor(
        None,
        lambda: codexvid_chat(
            body.query,
            hits,
            model=model,
            mode=mode,
            session_id=body.session_id,
        ),
    )
    return {
        "session_id": body.session_id,
        "answer": result["answer"],
        "timestamps": result["timestamps"],
        "timestamp_start": result.get("timestamp_start"),
        "timestamp_end": result.get("timestamp_end"),
        "key_points": result.get("key_points", []),
        "grounded": result.get("grounded"),
        "grounding_score": result.get("grounding_score"),
        "mode": result["mode"],
        "chunks_used": len(hits),
    }


@router.get("/sessions/{session_id}/exists")
async def codexvid_session_exists(session_id: str):
    ok = (CODEXVID_SESSIONS_DIR / session_id / "faiss.index").is_file()
    return {"session_id": session_id, "exists": ok}


@router.get("/sessions/{session_id}/video")
async def codexvid_session_video(session_id: str):
    """Original upload (for playback + timestamp seek). Copied into the session folder at process time."""
    base = CODEXVID_SESSIONS_DIR / session_id
    if not base.is_dir():
        return JSONResponse({"error": "Unknown session_id"}, status_code=404)
    for path in sorted(base.glob("source.*")):
        if path.is_file() and path.suffix.lower() in {
            ".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v",
        }:
            mime, _ = mimetypes.guess_type(str(path))
            return FileResponse(
                str(path),
                media_type=mime or "video/mp4",
                filename=path.name,
            )
    return JSONResponse({"error": "No video file for this session"}, status_code=404)
