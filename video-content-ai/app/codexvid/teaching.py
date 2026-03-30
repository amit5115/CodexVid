"""Per-chunk teaching pack: one LLM call per semantic chunk → aggregate → merge → coverage."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any

from app.codexvid.chunking import chunk_time_range
from app.config import CODEXVID_TEACHING_CHUNK_WORKERS
from app.core.llm import get_provider, normalize_llm_model_id

logger = logging.getLogger(__name__)

_MAX_TOPIC_DESC = 4_000
# Apply minimum-topic coverage enforcement for any video longer than 2 minutes.
_LONG_VIDEO_SEC = 120.0
_MIN_TOPICS_LONG_VIDEO = 5
_DEFAULT_MERGE_SIMILARITY = 0.90
_STRICT_MERGE_SIMILARITY = 0.98


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _extract_json(raw: str) -> dict | None:
    """Try multiple strategies to extract a JSON object from LLM output.

    Handles:
    - Plain JSON
    - Markdown fences (```json ... ```)
    - Preamble text before the object ("Here is the JSON: {...}")
    - Trailing text after the closing brace
    """
    text = _strip_json_fences(raw)

    # Strategy 1: direct parse after fence removal
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2: find outermost {...} block
    start = text.find("{")
    if start != -1:
        # Walk backwards from the end to find the matching closing brace
        end = text.rfind("}")
        if end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

    return None


def _sentence_boundary_times(sentences: list[dict]) -> list[float]:
    vals: list[float] = []
    for s in sentences:
        try:
            vals.append(float(s["start"]))
            vals.append(float(s["end"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(set(vals))


def _nearest_boundary(t: float, boundaries: list[float]) -> float:
    if not boundaries:
        return float(t)
    return float(min(boundaries, key=lambda b: abs(b - float(t))))


def snap_chapter_times_to_sentences(
    chapters: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Snap each chapter ``start``/``end`` to the nearest sentence boundary time."""
    bounds = _sentence_boundary_times(sentences)
    if not bounds:
        return chapters
    out: list[dict[str, Any]] = []
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        row = dict(ch)
        try:
            s0 = float(row["start"])
            s1 = float(row["end"])
        except (KeyError, TypeError, ValueError):
            out.append(row)
            continue
        ns0 = _nearest_boundary(s0, bounds)
        ns1 = _nearest_boundary(s1, bounds)
        if ns1 <= ns0:
            greater = [b for b in bounds if b > ns0]
            ns1 = float(min(greater)) if greater else ns0 + 10.0
        row["start"] = ns0
        row["end"] = ns1
        out.append(row)
    return out


def _fallback_topic(chunk: dict, a: float, b: float, title: str) -> dict[str, Any]:
    snippet = (chunk.get("text") or "").strip()
    if len(snippet) > 800:
        snippet = snippet[:800] + "…"
    return {
        "topic_title": title,
        "description": snippet or "No transcript text in this segment.",
        "start_time": float(a),
        "end_time": float(b),
    }


def _llm_topic_for_chunk(
    *,
    model: str,
    chunk: dict,
    chunk_index: int,
    total_chunks: int,
) -> dict[str, Any]:
    """Identify topic for ONE segment; times are taken from the chunk (ground truth)."""
    a, b = chunk_time_range(chunk)
    body = (chunk.get("text") or "").strip()
    if not body:
        return _fallback_topic(chunk, a, b, f"Segment {chunk_index + 1} (empty)")

    system = (
        "You are a precise segment analyzer. "
        "You receive ONE short transcript excerpt and must describe ONLY what is in that excerpt. "
        "Output ONLY valid JSON with no markdown fences, no preamble, no trailing text. "
        "NEVER describe topics from other parts of the video. "
        "NEVER summarize the entire video. "
        "NEVER use phrases like 'in this video', 'throughout the video', or 'the video covers'. "
        "Ground every sentence of your description in the provided transcript text."
    )
    user = f"""SEGMENT {chunk_index + 1} of {total_chunks}
TIME: {a:.3f}s – {b:.3f}s  (duration: {b - a:.1f}s)

TRANSCRIPT OF THIS SEGMENT ONLY:
\"\"\"
{body}
\"\"\"

YOUR TASK:
1. Read ONLY the transcript above.
2. Identify the single main topic being discussed in THIS segment.
3. Write a description (2–5 sentences) that covers what is said in THIS segment only.
4. Do NOT reference any other part of the video.
5. Do NOT write a general summary of the whole subject.

Return this JSON (fill in the blanks, keep start_time and end_time as the exact numbers given):
{{
  "topic_title": "<concise title for what THIS segment is about>",
  "description": "<2–5 sentences about THIS segment only>",
  "start_time": {a},
  "end_time": {b}
}}"""

    provider = get_provider(model=model)
    raw = provider.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    data = _extract_json(raw)
    if data is None:
        logger.warning(
            "Chunk %d: JSON parse failed (raw=%r…); fallback",
            chunk_index,
            raw[:120],
        )
        return _fallback_topic(chunk, a, b, f"Segment {chunk_index + 1}")

    title = str(data.get("topic_title") or "").strip() or f"Segment {chunk_index + 1}"
    desc = str(data.get("description") or "").strip() or body[:2000]
    if len(desc) > _MAX_TOPIC_DESC:
        desc = desc[:_MAX_TOPIC_DESC] + "…"

    # Detect phrases that indicate the LLM summarised the whole video instead of
    # the segment.  When this happens, substitute the raw transcript snippet so the
    # chunk at least covers its correct time range.
    _whole_video_phrases = (
        "in this video",
        "throughout the video",
        "the video covers",
        "the video discusses",
        "this video explores",
        "this video explains",
        "the entire video",
        "across the video",
    )
    if any(p in desc.lower() for p in _whole_video_phrases):
        logger.warning(
            "Chunk %d: LLM description appears to be a whole-video summary; using transcript snippet",
            chunk_index,
        )
        snippet = body[:600] + ("…" if len(body) > 600 else "")
        desc = snippet

    return {
        "topic_title": title,
        "description": desc,
        "start_time": float(a),
        "end_time": float(b),
    }


def _titles_similar(a: str, b: str, *, threshold: float) -> bool:
    x = re.sub(r"\s+", " ", (a or "").lower().strip())
    y = re.sub(r"\s+", " ", (b or "").lower().strip())
    if not x or not y:
        return False
    if x == y:
        return True
    return SequenceMatcher(None, x, y).ratio() >= threshold


def _consecutive_time(prev_end: float, next_start: float, eps: float = 2.0) -> bool:
    return abs(float(next_start) - float(prev_end)) <= eps


def merge_adjacent_topics(
    topics: list[dict[str, Any]],
    *,
    similarity_threshold: float = _DEFAULT_MERGE_SIMILARITY,
) -> list[dict[str, Any]]:
    """Merge adjacent topics only when titles are highly similar and times are consecutive."""
    if not topics:
        return []
    ordered = sorted(topics, key=lambda t: float(t["start_time"]))
    out: list[dict[str, Any]] = [dict(ordered[0])]
    for t in ordered[1:]:
        prev = out[-1]
        try:
            t0 = float(t["start_time"])
            t1 = float(t["end_time"])
            pe = float(prev["end_time"])
        except (KeyError, TypeError, ValueError):
            out.append(dict(t))
            continue
        if (
            _titles_similar(
                str(prev.get("topic_title", "")),
                str(t.get("topic_title", "")),
                threshold=similarity_threshold,
            )
            and _consecutive_time(pe, t0)
        ):
            prev["end_time"] = max(float(prev["end_time"]), t1)
            merged_desc = (
                str(prev.get("description", "")).strip()
                + "\n\n"
                + str(t.get("description", "")).strip()
            ).strip()
            if len(merged_desc) > _MAX_TOPIC_DESC:
                merged_desc = merged_desc[:_MAX_TOPIC_DESC] + "…"
            prev["description"] = merged_desc
        else:
            out.append(dict(t))
    return out


def _video_bounds(chunks: list[dict]) -> tuple[float, float]:
    t0, _ = chunk_time_range(chunks[0])
    _, t1 = chunk_time_range(chunks[-1])
    return float(t0), float(t1)


def enforce_coverage(
    topics: list[dict[str, Any]],
    chunks: list[dict],
) -> list[dict[str, Any]]:
    """Ensure first/last topic span matches full chunk timeline."""
    if not topics or not chunks:
        return topics
    t_min, t_max = _video_bounds(chunks)
    tps = sorted(topics, key=lambda x: float(x["start_time"]))
    tps[0]["start_time"] = min(float(tps[0]["start_time"]), t_min)
    tps[-1]["end_time"] = max(float(tps[-1]["end_time"]), t_max)
    return tps


def _topics_to_chapters(
    topics: list[dict[str, Any]],
    sentences: list[dict] | None,
) -> list[dict[str, Any]]:
    """Legacy chapter list for UI (title, start, end, explanation)."""
    rows: list[dict[str, Any]] = []
    for t in sorted(topics, key=lambda x: float(x["start_time"])):
        rows.append(
            {
                "title": str(t.get("topic_title") or "Topic"),
                "start": float(t["start_time"]),
                "end": float(t["end_time"]),
                "explanation": str(t.get("description") or ""),
            }
        )
    if sentences:
        rows = snap_chapter_times_to_sentences(rows, sentences)
    return rows


def _llm_takeaways_and_quiz(model: str, topic_summaries: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Second call: bullet summaries only (no full transcript)."""
    system = (
        "You create study aids from a list of segment topic summaries. "
        "Output ONLY valid JSON (no markdown fences). "
        "Do not invent facts not implied by the summaries."
    )
    user = f"""Chronological segment topics (from a video):

{topic_summaries}

Return JSON:
{{
  "key_takeaways": ["4 to 8 short bullets"],
  "quiz": [
    {{"question": "...", "answer": "short answer"}}
  ]
}}
Rules: exactly 3 quiz questions; 4–8 key takeaways."""

    provider = get_provider(model=model)
    raw = provider.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    data = _extract_json(raw)
    if data is None:
        logger.warning("Takeaways/quiz JSON parse failed (raw=%r…)", raw[:120])
        return [], []
    kt = data.get("key_takeaways") or []
    qz = data.get("quiz") or []
    if not isinstance(kt, list):
        kt = []
    if not isinstance(qz, list):
        qz = []
    out_q: list[dict[str, Any]] = [x for x in qz if isinstance(x, dict)]
    return [str(x) for x in kt if x], out_q


def generate_teaching_output(
    chunks: list[dict],
    *,
    model: str,
    sentences: list[dict] | None = None,
) -> dict[str, Any]:
    """One LLM call per semantic chunk (no full-transcript chapter generation), then aggregate.

    Returns ``chapters`` (UI), ``topics`` (structured), ``key_takeaways``, ``quiz``.
    """
    model = normalize_llm_model_id(model)
    if not chunks:
        return {
            "chapters": [],
            "topics": [],
            "key_takeaways": [],
            "quiz": [],
            "raw_error": "no_chunks",
        }

    n = len(chunks)
    workers = max(1, min(CODEXVID_TEACHING_CHUNK_WORKERS, n))

    def _run(i: int) -> tuple[int, dict[str, Any]]:
        try:
            c = chunks[i]
            topic = _llm_topic_for_chunk(
                model=model, chunk=c, chunk_index=i, total_chunks=n
            )
            return i, topic
        except Exception:
            logger.exception("Per-chunk teaching failed for index %d", i)
            a, b = chunk_time_range(chunks[i])
            return i, _fallback_topic(chunks[i], a, b, f"Segment {i + 1}")

    raw_topics: list[dict[str, Any] | None] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run, i) for i in range(n)]
        for fut in as_completed(futures):
            idx, topic = fut.result()
            raw_topics[idx] = topic

    topics_list: list[dict[str, Any]] = [t for t in raw_topics if t is not None]
    t_min, t_max = _video_bounds(chunks)
    duration = max(t_max - t_min, 0.0)

    merged = merge_adjacent_topics(topics_list, similarity_threshold=_DEFAULT_MERGE_SIMILARITY)

    if duration >= _LONG_VIDEO_SEC and len(merged) < _MIN_TOPICS_LONG_VIDEO and len(topics_list) >= _MIN_TOPICS_LONG_VIDEO:
        merged = merge_adjacent_topics(topics_list, similarity_threshold=_STRICT_MERGE_SIMILARITY)

    if duration >= _LONG_VIDEO_SEC and len(merged) < _MIN_TOPICS_LONG_VIDEO and len(topics_list) >= _MIN_TOPICS_LONG_VIDEO:
        merged = list(
            sorted(topics_list, key=lambda x: float(x["start_time"]))
        )

    merged = enforce_coverage(merged, chunks)

    summary_lines = [
        f"- [{float(t['start_time']):.1f}-{float(t['end_time']):.1f}s] {t.get('topic_title', '')}: "
        f"{str(t.get('description', ''))[:220]}"
        for t in sorted(merged, key=lambda x: float(x["start_time"]))
    ]
    summary_blob = "\n".join(summary_lines)[:12_000]

    key_takeaways, quiz = _llm_takeaways_and_quiz(model, summary_blob)

    chapters = _topics_to_chapters(merged, sentences if sentences else None)

    return {
        "chapters": chapters,
        "topics": merged,
        "key_takeaways": key_takeaways,
        "quiz": quiz,
    }
