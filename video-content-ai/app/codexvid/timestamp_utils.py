"""Timestamp alignment, merging, and normalization for CodexVid transcripts and chapters."""

from __future__ import annotations

import re
from typing import Any

# Sentence-ending punctuation for boundary hints
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# No lead-in pad: keeps transcript times aligned with video seek (exact second jumps).
_LEAD_PAD_SEC = 0.0
_MIN_SEG_SEC = 0.2
_MIN_WORD_GAP = 0.02
_DEDUPE_TIME = 0.18


def merge_segments(segments: list[dict]) -> list[dict]:
    """Merge overlapping / duplicate Whisper segments; enforce chronological order.

    Call after parallel overlapping windows or duplicate-prone paths.
    """
    if not segments:
        return []

    rows: list[dict] = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        a = float(s["start"])
        b = float(s["end"])
        if b <= a:
            b = a + _MIN_SEG_SEC
        rows.append(
            {
                "text": text,
                "start": a,
                "end": b,
                "words": s.get("words"),
            }
        )

    rows.sort(key=lambda x: (x["start"], x["end"]))

    deduped: list[dict] = []
    for s in rows:
        if deduped and abs(s["start"] - deduped[-1]["start"]) < 0.4 and s["text"] == deduped[-1]["text"]:
            deduped[-1]["end"] = max(deduped[-1]["end"], s["end"])
            continue
        deduped.append(dict(s))

    out: list[dict] = []
    for s in deduped:
        if not out:
            out.append(s)
            continue
        p = out[-1]
        if s["start"] >= p["end"] - _MIN_WORD_GAP:
            out.append(dict(s))
            continue
        if s["text"] == p["text"]:
            p["end"] = max(p["end"], s["end"])
            p["words"] = s.get("words") or p.get("words")
            continue
        if s["text"] in p["text"]:
            p["end"] = max(p["end"], s["end"])
            continue
        if p["text"] in s["text"]:
            out[-1] = dict(s)
            continue
        shifted = dict(s)
        shifted["start"] = max(float(s["start"]), float(p["end"]) + _MIN_WORD_GAP)
        if shifted["end"] <= shifted["start"]:
            shifted["end"] = shifted["start"] + _MIN_SEG_SEC
        if shifted["end"] > shifted["start"] + _MIN_WORD_GAP:
            out.append(shifted)
    return out


def dedupe_overlapping_words(words: list[dict]) -> list[dict]:
    """Remove duplicate words caused by overlapping audio windows."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (w["start"], w["end"]))
    out: list[dict] = []
    for w in words:
        lw = (w.get("word") or "").strip()
        if not lw:
            continue
        if out:
            o = out[-1]
            if (
                abs(float(w["start"]) - float(o["start"])) < _DEDUPE_TIME
                and lw.lower() == (o.get("word") or "").strip().lower()
            ):
                out[-1]["end"] = max(float(o["end"]), float(w["end"]))
                continue
        out.append(
            {
                "word": lw,
                "start": float(w["start"]),
                "end": float(w["end"]),
            }
        )
    return out


def words_to_fine_segments(
    words: list[dict],
    *,
    min_sec: float = 2.0,
    max_sec: float = 5.0,
) -> list[dict]:
    """Group word timings into 2–5s segments with natural breaks when possible."""
    if not words:
        return []
    wk = dedupe_overlapping_words(words)
    if not wk:
        return []

    segs: list[dict] = []
    i = 0
    n = len(wk)
    while i < n:
        t0 = float(wk[i]["start"])
        j = i
        t1 = float(wk[i]["end"])
        while j < n - 1:
            next_end = float(wk[j + 1]["end"])
            cand_dur = next_end - t0
            if cand_dur > max_sec:
                break
            j += 1
            t1 = next_end
            dur = t1 - t0
            wtext = (wk[j].get("word") or "").strip()
            if dur >= min_sec and re.search(r"[.!?…]$", wtext):
                break
            if dur >= max_sec:
                break
        chunk_words = [
            {"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
            for w in wk[i : j + 1]
        ]
        text = " ".join(str(w["word"]) for w in chunk_words).strip()
        if text:
            segs.append(
                {
                    "text": text,
                    "start": round(t0, 3),
                    "end": round(t1, 3),
                    "words": chunk_words,
                }
            )
        i = j + 1

    return segs


_SENTENCE_END_RE = re.compile(r"[.!?…]['\"]?\s*$")


def flatten_words_from_transcript(transcript: list[dict]) -> list[dict]:
    """Collect word-level timings from segments (Whisper ``words``) or linear interpolation."""
    words: list[dict] = []
    for seg in transcript:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        seg_words_in = seg.get("words")

        if seg_words_in and isinstance(seg_words_in, list) and len(seg_words_in) > 0:
            for w in seg_words_in:
                if not isinstance(w, dict):
                    continue
                wt = str(w.get("word") or "").strip()
                if not wt:
                    continue
                words.append(
                    {
                        "word": wt,
                        "start": float(w.get("start", start)),
                        "end": float(w.get("end", end)),
                    }
                )
            continue

        seg_words = text.split()
        if not seg_words:
            continue
        dur = max(end - start, 1e-3)
        n = len(seg_words)
        for i, w in enumerate(seg_words):
            t0 = start + (i / n) * dur
            t1 = start + ((i + 1) / n) * dur
            words.append({"word": w, "start": t0, "end": t1})

    words.sort(key=lambda x: (float(x["start"]), float(x["end"])))
    return words


def words_to_sentence_spans(words: list[dict]) -> list[dict]:
    """Group words into sentences using ending punctuation; each has ``text``, ``start``, ``end``, ``words``."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (float(w["start"]), float(w["end"])))
    out: list[dict] = []
    cur: list[dict] = []
    for w in words:
        wt = (w.get("word") or "").strip()
        if not wt:
            continue
        cur.append({"word": wt, "start": float(w["start"]), "end": float(w["end"])})
        stripped = wt.rstrip("\"'”’")
        if _SENTENCE_END_RE.search(stripped) or stripped.endswith("…"):
            text = " ".join(x["word"] for x in cur).strip()
            if text:
                out.append(
                    {
                        "text": text,
                        "start": float(cur[0]["start"]),
                        "end": float(cur[-1]["end"]),
                        "words": [dict(x) for x in cur],
                    }
                )
            cur = []
    if cur:
        text = " ".join(x["word"] for x in cur).strip()
        if text:
            out.append(
                {
                    "text": text,
                    "start": float(cur[0]["start"]),
                    "end": float(cur[-1]["end"]),
                    "words": [dict(x) for x in cur],
                }
            )
    return out


def transcript_sentence_timeline(transcript: list[dict]) -> list[dict]:
    """Full-video sentence list ``{text, start, end}`` from word-level data."""
    words = flatten_words_from_transcript(transcript)
    if not words:
        return []
    spans = words_to_sentence_spans(words)
    return [{"text": s["text"], "start": s["start"], "end": s["end"]} for s in spans]


def align_timestamps(segments: list[dict]) -> list[dict]:
    """Refine start/end: snap to words when present, small lead-in pad, fix ordering."""
    if not segments:
        return []

    out: list[dict] = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        words = s.get("words")
        if words and isinstance(words, list) and len(words) > 0:
            ws = sorted(words, key=lambda x: float(x["start"]))
            a = float(ws[0]["start"])
            b = float(ws[-1]["end"])
        else:
            a = float(s["start"])
            b = float(s["end"])

        # Slight earlier start for seek UX (hear lead-in)
        a = max(0.0, a - _LEAD_PAD_SEC)
        if b <= a:
            b = a + _MIN_SEG_SEC

        out.append({"text": text, "start": round(a, 3), "end": round(b, 3), "words": words})

    out.sort(key=lambda x: (x["start"], x["end"]))

    # Second pass: enforce monotonic ends and small gaps
    fixed: list[dict] = []
    for i, s in enumerate(out):
        a, b = float(s["start"]), float(s["end"])
        if fixed:
            prev = fixed[-1]
            pa, pb = float(prev["start"]), float(prev["end"])
            if a < pb:
                a = pb + _MIN_WORD_GAP
            if b <= a:
                b = a + _MIN_SEG_SEC
        fixed.append({**s, "start": round(a, 3), "end": round(b, 3)})
    return fixed


def normalize_transcript_segments(segments: list[dict]) -> list[dict]:
    """merge_segments → align_timestamps → drop ultra-short noise."""
    merged = merge_segments(segments)
    aligned = align_timestamps(merged)
    # Remove very short noisy segments (< min word count and tiny duration)
    filtered: list[dict] = []
    for s in aligned:
        dur = float(s["end"]) - float(s["start"])
        wc = len((s.get("text") or "").split())
        if dur < 0.12 and wc < 2:
            continue
        filtered.append(s)
    return filtered


def clean_timestamps(chapters: list[dict]) -> list[dict]:
    """Post-process LLM chapters: sort, remove overlaps, merge short segments, normalize gaps.

    Returns list of ``{"title", "start", "end"}`` with numeric seconds.
    """
    if not chapters:
        return []

    rows: list[dict[str, Any]] = []
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        title = (ch.get("title") or "Section").strip() or "Section"
        try:
            start = float(ch.get("start", 0))
            end = float(ch.get("end", 0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            end = start + 10.0
        rows.append({"title": title, "start": start, "end": end})

    if not rows:
        return []

    rows.sort(key=lambda x: (x["start"], x["end"]))

    merged: list[dict[str, Any]] = []
    MIN_CHAPTER = 10.0
    GAP_MIN = 0.5

    for r in rows:
        if not merged:
            merged.append(dict(r))
            continue
        p = merged[-1]
        # Overlap resolution
        if r["start"] < p["end"]:
            if r["end"] <= p["end"] + 0.01:
                continue
            r = {**r, "start": p["end"] + GAP_MIN}
        if r["end"] <= r["start"]:
            r["end"] = r["start"] + MIN_CHAPTER

        dur = r["end"] - r["start"]
        if dur < MIN_CHAPTER and merged:
            # Merge small segment into previous
            p["title"] = f"{p['title']} / {r['title']}"[:200]
            p["end"] = max(p["end"], r["end"])
            continue

        merged.append(r)

    # Normalize gaps: ensure end[i] < start[i+1] or touch
    out: list[dict[str, float | str]] = []
    for i, r in enumerate(merged):
        start = float(r["start"])
        end = float(r["end"])
        if out:
            prev_end = float(out[-1]["end"])
            if start < prev_end + GAP_MIN:
                start = prev_end + GAP_MIN
            if end <= start:
                end = start + MIN_CHAPTER
        out.append({"title": str(r["title"]), "start": start, "end": end})

    return out  # type: ignore[return-value]
