"""Sentence-level retrieval helpers: batch embeddings + cosine match within chunk windows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.codexvid.chunking import chunk_time_range
from app.config import CODEXVID_SESSIONS_DIR, EMBEDDING_MODEL
from app.core.llm import get_provider

logger = logging.getLogger(__name__)


def embed_texts(texts: list[str], *, model: str | None = None) -> np.ndarray:
    """Return embedding matrix ``(n, dim)`` float32 via the configured LLM provider.

    One batch call per invocation — callers should pass all texts together, not loop.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    m = model or EMBEDDING_MODEL
    provider = get_provider()
    vecs = provider.embed(model=m, texts=texts)
    if not vecs or len(vecs) != len(texts):
        raise RuntimeError("embedding length mismatch")
    return np.asarray(vecs, dtype=np.float32)


def cosine_similarity_matrix(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between query row(s) and each row of ``matrix``.

    ``q`` shape ``(d,)`` or ``(1, d)``; ``matrix`` shape ``(n, d)``. Returns ``(n,)``.
    """
    qv = np.asarray(q, dtype=np.float32).reshape(-1)
    mv = np.asarray(matrix, dtype=np.float32)
    if mv.size == 0:
        return np.array([], dtype=np.float32)
    n = mv.shape[0]
    qn = float(np.linalg.norm(qv))
    if qn < 1e-12:
        return np.zeros(n, dtype=np.float32)
    qv = qv / qn
    mn = np.linalg.norm(mv, axis=1, keepdims=True)
    mn = np.maximum(mn, 1e-12)
    mv = mv / mn
    return (mv @ qv).reshape(-1)


def filter_sentences_overlapping_chunks(
    sentences: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sentences that overlap any retrieved chunk time range (interval intersection)."""
    if not sentences or not chunks:
        return []
    out: list[dict[str, Any]] = []
    for s in sentences:
        try:
            sa = float(s["start"])
            sb = float(s["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if sb <= sa:
            continue
        for c in chunks:
            ca, cb = chunk_time_range(c)
            if sa < cb and sb > ca:
                out.append(s)
                break
    return out


def find_most_relevant_sentence(
    query: str,
    sentences: list[dict[str, Any]],
    *,
    embed_model: str | None = None,
) -> dict[str, Any] | None:
    """Return the single sentence dict most similar to ``query`` (one batched embed call)."""
    q = (query or "").strip()
    if not q or not sentences:
        return None

    kept: list[dict[str, Any]] = []
    for s in sentences:
        t = str(s.get("text") or "").strip()
        if t:
            kept.append(s)
    if not kept:
        return None

    texts = [q] + [str(s.get("text") or "").strip() for s in kept]
    batch = embed_texts(texts, model=embed_model)
    if batch.shape[0] != len(texts):
        logger.warning("embedding batch size mismatch; skipping sentence pick")
        return None

    qv = batch[0]
    sv = batch[1:]
    sims = cosine_similarity_matrix(qv, sv)
    if sims.size == 0:
        return None
    best_i = int(np.argmax(sims))
    return kept[best_i]


def load_session_sentences(session_id: str) -> list[dict[str, Any]]:
    """Load ``sentences`` from ``transcript.json``; support legacy list-only transcripts."""
    path = Path(CODEXVID_SESSIONS_DIR) / session_id / "transcript.json"
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Could not read transcript for %s: %s", session_id, e)
        return []

    if isinstance(raw, dict):
        sents = raw.get("sentences")
        if isinstance(sents, list) and sents:
            return [x for x in sents if isinstance(x, dict)]
        segs = raw.get("segments")
        if isinstance(segs, list) and segs:
            from app.codexvid.timestamp_utils import transcript_sentence_timeline

            return transcript_sentence_timeline(segs)
        return []

    if isinstance(raw, list):
        from app.codexvid.timestamp_utils import transcript_sentence_timeline

        return transcript_sentence_timeline(raw)

    return []
