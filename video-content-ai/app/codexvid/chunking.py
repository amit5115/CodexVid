"""Semantic transcript chunks with word-level timing: 30–60s ideas, sentence boundaries."""

from __future__ import annotations

import logging

from app.config import CODEXVID_SEM_CHUNK_MAX_SEC, CODEXVID_SEM_CHUNK_MIN_SEC
from app.codexvid.timestamp_utils import (
    flatten_words_from_transcript,
    words_to_sentence_spans,
)

logger = logging.getLogger(__name__)


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
    """When one sentence exceeds ``max_sec``, split on word timings.

    Falls back to a single chunk when word-level data is unavailable.  Previously
    this returned ``[]`` for a no-words oversized sentence, silently dropping the
    entire segment from the output.
    """
    words = sent.get("words") or []
    if not words:
        # No word-level data: produce one chunk covering the whole sentence.
        # The caller (_pack_sentences) tolerates chunks > max_sec when there is no
        # finer boundary to split on.
        return [
            {
                "text": str(sent.get("text") or "").strip(),
                "start_time": float(sent["start"]),
                "end_time": float(sent["end"]),
            }
        ]
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


def _chunk_segments_by_time(
    segments: list[dict],
    *,
    min_sec: float,
    max_sec: float,
) -> list[dict]:
    """Direct time-based chunking from segment start/end timestamps.

    This is the most robust fallback: it requires nothing beyond the segment's
    ``start``/``end`` (or ``start_time``/``end_time``) and ``text`` fields.
    Used when word-based or sentence-based chunking produces too few chunks.
    """
    if not segments:
        return []

    def _seg_start(s: dict) -> float:
        return float(s.get("start", s.get("start_time", 0.0)))

    def _seg_end(s: dict) -> float:
        return float(s.get("end", s.get("end_time", _seg_start(s) + 0.1)))

    ordered = sorted(segments, key=_seg_start)
    chunks: list[dict] = []
    acc_texts: list[str] = []
    acc_start: float | None = None
    acc_end: float = 0.0

    for seg in ordered:
        s0 = _seg_start(seg)
        s1 = _seg_end(seg)
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        if acc_start is None:
            acc_start = s0
            acc_end = s1
            acc_texts = [text]
            continue

        dur_if_add = s1 - acc_start
        if dur_if_add > max_sec and acc_texts:
            # Flush current accumulation before adding this segment.
            chunks.append({
                "text": " ".join(acc_texts),
                "start_time": acc_start,
                "end_time": acc_end,
            })
            acc_start = s0
            acc_end = s1
            acc_texts = [text]
        else:
            acc_texts.append(text)
            acc_end = s1
            # Flush once we reach min_sec.
            if s1 - acc_start >= min_sec:
                chunks.append({
                    "text": " ".join(acc_texts),
                    "start_time": acc_start,
                    "end_time": acc_end,
                })
                acc_start = None
                acc_texts = []

    # Flush any remainder.
    if acc_texts and acc_start is not None:
        if chunks:
            tail_dur = acc_end - acc_start
            # Merge a tiny tail (< half min_sec) into the last chunk.
            if tail_dur < min_sec * 0.5:
                chunks[-1]["text"] = (chunks[-1]["text"] + " " + " ".join(acc_texts)).strip()
                chunks[-1]["end_time"] = acc_end
            else:
                chunks.append({
                    "text": " ".join(acc_texts),
                    "start_time": acc_start,
                    "end_time": acc_end,
                })
        else:
            chunks.append({
                "text": " ".join(acc_texts),
                "start_time": acc_start,
                "end_time": acc_end,
            })

    return chunks


def _expected_min_chunks(video_duration_sec: float, max_sec: float) -> int:
    """Minimum acceptable chunk count for a given video duration."""
    return max(1, int(video_duration_sec / max_sec))


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

    **Robustness guarantees:**
    - If sentence-based chunking produces fewer chunks than expected for the video
      duration, the function falls back to direct segment-time chunking.
    - If words have corrupted / clustered-at-zero timestamps, the fallback still
      produces correct chunks because it uses raw segment start/end times.
    """
    del chunk_size, overlap
    min_sec = CODEXVID_SEM_CHUNK_MIN_SEC
    max_sec = CODEXVID_SEM_CHUNK_MAX_SEC
    if min_sec > max_sec:
        min_sec, max_sec = max_sec, min_sec

    # ------------------------------------------------------------------
    # Step 1: word-based / sentence-based chunking (preferred path)
    # ------------------------------------------------------------------
    words = flatten_words_from_transcript(transcript)

    packed: list[dict] = []
    if words:
        sentences = words_to_sentence_spans(words)
        if sentences:
            packed = _pack_sentences(sentences, min_sec=min_sec, max_sec=max_sec)
        if not packed:
            packed = _chunk_words_by_max_duration(
                [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in words],
                max_sec=max_sec,
            )

    # ------------------------------------------------------------------
    # Step 2: compute video duration from transcript for validation
    # ------------------------------------------------------------------
    if transcript:
        all_starts = [
            float(s.get("start", s.get("start_time", 0.0))) for s in transcript
        ]
        all_ends = [
            float(s.get("end", s.get("end_time", 0.0))) for s in transcript
        ]
        video_duration = max(all_ends) - min(all_starts) if all_ends and all_starts else 0.0
    else:
        video_duration = 0.0

    min_expected = _expected_min_chunks(video_duration, max_sec)

    # ------------------------------------------------------------------
    # Step 3: fallback to direct segment-time chunking when the word-based
    # result has too few chunks (e.g. because word timestamps were corrupted
    # or all words clustered near t=0 due to missing Whisper word_timestamps).
    # ------------------------------------------------------------------
    if len(packed) < min_expected:
        time_based = _chunk_segments_by_time(transcript, min_sec=min_sec, max_sec=max_sec)
        if len(time_based) >= len(packed):
            logger.info(
                "create_chunks: word-based produced %d chunk(s) (expected ≥%d for %.1fs video); "
                "using time-based fallback (%d chunks)",
                len(packed), min_expected, video_duration, len(time_based),
            )
            packed = time_based

    if not packed:
        # Last resort: one chunk covering the entire transcript.
        if transcript:
            full_text = " ".join(
                (s.get("text") or "").strip() for s in transcript if (s.get("text") or "").strip()
            )
            all_starts2 = [float(s.get("start", s.get("start_time", 0.0))) for s in transcript]
            all_ends2 = [float(s.get("end", s.get("end_time", 0.0))) for s in transcript]
            if full_text:
                packed = [{
                    "text": full_text,
                    "start_time": min(all_starts2),
                    "end_time": max(all_ends2),
                }]

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
