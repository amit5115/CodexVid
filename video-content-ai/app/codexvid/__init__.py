"""CodexVid pipeline: transcribe → chunk → FAISS → chat (timestamp-grounded)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.codexvid.chat import chat, detect_mode
from app.codexvid.chunking import create_chunks
from app.codexvid.teaching import generate_teaching_output
from app.codexvid.transcription import transcribe_video

if TYPE_CHECKING:
    from app.codexvid.vector_store import CodexvidVectorStore


def __getattr__(name: str):
    if name == "CodexvidVectorStore":
        from app.codexvid.vector_store import CodexvidVectorStore as _T

        return _T
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "chat",
    "create_chunks",
    "detect_mode",
    "generate_teaching_output",
    "CodexvidVectorStore",
    "transcribe_video",
]
