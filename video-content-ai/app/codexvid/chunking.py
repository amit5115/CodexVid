"""Semantic transcript chunks with word-level timing: 30–60s ideas, sentence boundaries."""

from __future__ import annotations

from app.config import CODEXVID_SEM_CHUNK_MAX_SEC, CODEXVID_SEM_CHUNK_MIN_SEC
from app.codexvid.timestamp_utils import (
    flatten_words_from_transcript,
    words_to_sentence_spans,
)


def _chunk_words_by_max_duration(
    words: list[dict],
    *,
    max_sec: float,
) -> list[dict]:
    """Split a long word sequence into sub-chunks of at most ``max_sec`` duration."""
    if not words:
        return []
    out: list[dict] = []
    i = 0
    n = len(words)
    while i < n:
        t0 = float(words[i]["start"])
        j = i
        t_end = float(words[i]["end"])
        while j + 1 < n:
            t_next = float(words[j + 1]["end"])
            if t_next - t0 - 1e-6 > max_sec:
                break
            j += 1
            t_end = t_next
        piece = words[i : j + 1]
        text = " ".join(str(w["word"]) for w in piece).strip()
        if text:
            out.append(
                {
                    "text": text,
                    "start_time": float(piece[0]["start"]),
                    "end_time": float(piece[-1]["end"]),
                }
            )
        i = j + 1
    return out


def _split_oversized_sentence(sent: dict, *, max_sec: float) -> list[dict]:
    """When one sentence exceeds ``max_sec``, split on word timings."""
    words = sent.get("words") or []
    if not words:
        dur = float(sent["end"]) - float(sent["start"])
        if dur <= max_sec:
            return [
                {
                    "text": sent["text"],
                    "start_time": float(sent["start"]),
                    "end_time": float(sent["end"]),
                }
            ]
        return []
    return _chunk_words_by_max_duration(
        [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in words],
        max_sec=max_sec,
    )


def _pack_sentences(
    sentences: list[dict],
    *,
    min_sec: float,
    max_sec: float,
) -> list[dict]:
    """Greedy pack into semantic chunks (single topic window, 30–60s when possible)."""
    if not sentences:
        return []
    chunks: list[dict] = []
    i = 0
    n = len(sentences)
    while i < n:
        s0 = sentences[i]
        dur0 = float(s0["end"]) - float(s0["start"])
        if dur0 > max_sec:
            chunks.extend(_split_oversized_sentence(s0, max_sec=max_sec))
            i += 1
            continue

        acc = [s0]
        j = i + 1
        while j < n:
            cand = acc + [sentences[j]]
            t_start = float(cand[0]["start"])
            t_end = float(cand[-1]["end"])
            dur = t_end - t_start
            if dur > max_sec:
                break
            acc.append(sentences[j])
            j += 1
            if dur >= max_sec:
                break

        # Extend short chunks toward ≥ min_sec without exceeding max_sec
        while j < n:
            t_start = float(acc[0]["start"])
            t_end = float(sentences[j]["end"])
            dur_if_add = t_end - t_start
            if dur_if_add > max_sec:
                break
            acc.append(sentences[j])
            j += 1
            if dur_if_add >= min_sec:
                break

        text = " ".join(s["text"] for s in acc).strip()
        chunks.append(
            {
                "text": text,
                "start_time": float(acc[0]["start"]),
                "end_time": float(acc[-1]["end"]),
            }
        )
        i = j

    return chunks


def create_chunks(
    transcript: list[dict],
    *,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict]:
    """Build semantic chunks from word-level transcript data.

    Each chunk has ``text``, ``start_time``, ``end_time`` (seconds, floats for exact seek).
    Legacy parameters ``chunk_size`` / ``overlap`` are ignored (fixed windows removed).

    Aliases ``start`` / ``end`` mirror ``start_time`` / ``end_time`` for older callers.
    """
    del chunk_size, overlap
    min_sec = CODEXVID_SEM_CHUNK_MIN_SEC
    max_sec = CODEXVID_SEM_CHUNK_MAX_SEC
    if min_sec > max_sec:
        min_sec, max_sec = max_sec, min_sec

    words = flatten_words_from_transcript(transcript)
    if not words:
        return []

    sentences = words_to_sentence_spans(words)
    if not sentences:
        raw = _chunk_words_by_max_duration(
            [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in words],
            max_sec=max_sec,
        )
        return [_with_aliases(c) for c in raw]

    packed = _pack_sentences(sentences, min_sec=min_sec, max_sec=max_sec)
    return [_with_aliases(c) for c in packed]


def _with_aliases(c: dict) -> dict:
    d = dict(c)
    d["start"] = float(d["start_time"])
    d["end"] = float(d["end_time"])
    return d


def chunk_time_range(c: dict) -> tuple[float, float]:
    """Return (start, end) seconds for a chunk dict (supports legacy or new keys)."""
    if "start_time" in c and "end_time" in c:
        return float(c["start_time"]), float(c["end_time"])
    return float(c["start"]), float(c["end"])
