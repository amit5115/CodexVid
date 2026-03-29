"""Load/save CodexVid session artifacts (JSON + FAISS)."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from app.config import CODEXVID_SESSIONS_DIR
from app.codexvid.chunking import create_chunks
from app.codexvid.teaching import generate_teaching_output
from app.codexvid.timestamp_utils import transcript_sentence_timeline
from app.codexvid.transcription import transcribe_video
from app.codexvid.vector_store import CodexvidVectorStore

_VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def new_session_dir() -> tuple[str, Path]:
    sid = uuid.uuid4().hex
    path = CODEXVID_SESSIONS_DIR / sid
    path.mkdir(parents=True, exist_ok=True)
    return sid, path


def process_upload(
    video_path: Path,
    *,
    whisper_model: str = "base",
    language: str = "en",
    llm_model: str,
) -> tuple[str, dict]:
    """Full pipeline: transcribe → chunk → index → teaching JSON. Returns (session_id, payload)."""
    sid, session_dir = new_session_dir()
    ext = video_path.suffix.lower() or ".mp4"
    if ext not in _VIDEO_EXT:
        ext = ".mp4"
    shutil.copy2(video_path, session_dir / f"source{ext}")

    segments = transcribe_video(video_path, model_size=whisper_model, language=language)
    sentences = transcript_sentence_timeline(segments)
    (session_dir / "transcript.json").write_text(
        json.dumps(
            {"segments": segments, "sentences": sentences},
            indent=2,
        ),
        encoding="utf-8",
    )

    chunks = create_chunks(segments)
    (session_dir / "chunks.json").write_text(
        json.dumps(chunks, indent=2), encoding="utf-8"
    )

    if chunks:
        store = CodexvidVectorStore.build(chunks, session_dir)
        store.save()
    else:
        # Without an index, load_store() fails and chat returns 404 — always persist FAISS files.
        store = CodexvidVectorStore.build_empty(session_dir)
        store.save()

    teaching = generate_teaching_output(
        chunks,
        model=llm_model,
        sentences=sentences if sentences else None,
    )
    (session_dir / "teaching.json").write_text(
        json.dumps(teaching, indent=2), encoding="utf-8"
    )

    return sid, {
        "session_id": sid,
        "segment_count": len(segments),
        "chunk_count": len(chunks),
        "teaching": teaching,
    }


def load_store(session_id: str) -> CodexvidVectorStore:
    session_dir = CODEXVID_SESSIONS_DIR / session_id
    if not session_dir.is_dir() or not (session_dir / "faiss.index").is_file():
        raise FileNotFoundError(session_id)
    return CodexvidVectorStore.load(session_dir)
