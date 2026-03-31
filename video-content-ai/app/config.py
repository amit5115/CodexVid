"""CodexVid AI — application settings loaded from environment / .env file."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env_file() -> None:
    path = BASE_DIR / ".env"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if not key:
            continue
        cur = os.environ.get(key)
        if cur is None or (isinstance(cur, str) and not cur.strip()):
            os.environ[key] = val


_load_env_file()

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("VCAI_DATA_DIR", str(BASE_DIR / "data")))
STATIC_DIR = Path(__file__).resolve().parent / "static"
CODEXVID_SESSIONS_DIR = Path(
    os.getenv(
        "VCAI_CODEXVID_SESSIONS_DIR",
        os.getenv("VCAI_TEACHER_SESSIONS_DIR", str(DATA_DIR / "codexvid_sessions")),
    )
)

# ── Server ─────────────────────────────────────────────────────────────────────
HOST = os.getenv("VCAI_HOST", "0.0.0.0")
PORT = int(os.getenv("VCAI_PORT", "8501"))
RELOAD = os.getenv("VCAI_RELOAD", "true").lower() in ("1", "true", "yes")

# ── LLM / Embeddings ───────────────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("VCAI_DEFAULT_MODEL", "llama3")
EMBEDDING_MODEL = os.getenv("VCAI_EMBEDDING_MODEL", "nomic-embed-text")

# Company GPT internal proxy (used when model id matches gpt-4.x/gpt-4o variants)
COMPANY_GPT_ENDPOINT = os.getenv("COMPANY_GPT_ENDPOINT", "https://ai-framework1:8085")
COMPANY_GPT_API_KEY = os.getenv("COMPANY_GPT_API_KEY", "")
COMPANY_GPT_CALLER = os.getenv("COMPANY_GPT_CALLER", "")

# ── Speech-to-Text ─────────────────────────────────────────────────────────────
STT_PROVIDER = os.getenv("VCAI_STT_PROVIDER", "whisper").strip().lower()
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
AWS_TRANSCRIBE_BUCKET = os.getenv("VCAI_AWS_TRANSCRIBE_BUCKET", "").strip()
AWS_TRANSCRIBE_POLL_TIMEOUT_SEC = int(os.getenv("VCAI_AWS_TRANSCRIBE_POLL_TIMEOUT_SEC", "3600"))

# ── services/transcription.py (general Whisper helpers) ───────────────────────
CHUNK_DURATION_SEC = int(os.getenv("VCAI_CHUNK_DURATION", "300"))
MIN_DURATION_FOR_CHUNKING = int(os.getenv("VCAI_MIN_CHUNK_DURATION", "600"))
STREAM_CHUNK_DURATION_SEC = int(os.getenv("VCAI_STREAM_CHUNK_DURATION", "45"))
EARLY_CHAT_CHUNKS = int(os.getenv("VCAI_EARLY_CHAT_CHUNKS", "2"))
PARALLEL_TRANSCRIPTION_WORKERS = int(os.getenv("VCAI_PARALLEL_WORKERS", "2"))

# ── CodexVid pipeline tuning ───────────────────────────────────────────────────
# Whisper audio windows
CODEXVID_WHISPER_CHUNK_SEC = int(
    os.getenv("VCAI_CODEXVID_CHUNK_SEC") or os.getenv("VCAI_TEACHER_CHUNK_SEC") or "25"
)
CODEXVID_PARALLEL_WORKERS = int(
    os.getenv("VCAI_CODEXVID_PARALLEL_WORKERS") or os.getenv("VCAI_TEACHER_PARALLEL_WORKERS") or "4"
)
# Overlap between audio windows (reduces Whisper boundary drops); step = chunk - overlap
CODEXVID_AUDIO_OVERLAP_SEC = float(
    os.getenv("VCAI_CODEXVID_AUDIO_OVERLAP_SEC") or os.getenv("VCAI_CODEXVID_OVERLAP_SEC") or "5"
)
# Fine segments built from word timestamps (2–5 s each)
CODEXVID_FINE_SEG_MIN_SEC = float(os.getenv("VCAI_CODEXVID_FINE_SEG_MIN_SEC") or "2")
CODEXVID_FINE_SEG_MAX_SEC = float(os.getenv("VCAI_CODEXVID_FINE_SEG_MAX_SEC") or "5")

# RAG: top-k chunks retrieved per chat query
CODEXVID_RAG_TOP_K = int(os.getenv("VCAI_CODEXVID_RAG_TOP_K") or "3")

# Semantic chunks: one topic per window, sentence-boundary-aware
CODEXVID_SEM_CHUNK_MIN_SEC = float(os.getenv("VCAI_CODEXVID_SEM_CHUNK_MIN_SEC") or "30")
CODEXVID_SEM_CHUNK_MAX_SEC = float(os.getenv("VCAI_CODEXVID_SEM_CHUNK_MAX_SEC") or "60")

# Per-chunk teaching: parallel LLM calls (capped to avoid overloading local Ollama)
CODEXVID_TEACHING_CHUNK_WORKERS = int(os.getenv("VCAI_CODEXVID_TEACHING_CHUNK_WORKERS") or "4")


def ensure_dirs() -> None:
    for d in (DATA_DIR, CODEXVID_SESSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
