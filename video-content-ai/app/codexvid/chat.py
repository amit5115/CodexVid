"""Multi-stage CodexVid chat: extract → explain, JSON timestamps, grounding checks."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.codexvid.chunking import chunk_time_range
from app.codexvid.retrieval_utils import (
    filter_sentences_overlapping_chunks,
    find_most_relevant_sentence,
    load_session_sentences,
)
from app.core.llm import get_provider, normalize_llm_model_id

logger = logging.getLogger(__name__)

NOT_IN_VIDEO = "This is not covered in the video"
LOW_CONFIDENCE = "Not clearly explained in this segment"
NOT_MENTIONED = "Not mentioned in the video"

_MODE_HINTS = {
    "simple": (
        "Use very simple words and short sentences. Assume the learner is stuck; avoid jargon."
    ),
    "detailed": (
        "Give a thorough, structured explanation with clear sections. Include nuance where the transcript supports it."
    ),
    "analogy": (
        "Start with a concrete analogy that maps to the idea, then tie it back to what the video says."
    ),
    "example": (
        "Lead with a concrete example grounded in the context; then explain how it connects."
    ),
}


def _mmss(seconds: float) -> str:
    s = max(0.0, float(seconds))
    m = int(s // 60)
    sec = s - m * 60
    return f"{m:02d}:{sec:05.2f}"


def mmss_label_to_seconds(label: str) -> float:
    """Parse ``mm:ss`` or ``h:mm:ss`` → seconds (for video seek)."""
    parts = label.strip().split(":")
    try:
        if len(parts) == 2:
            m, s = int(parts[0]), float(parts[1])
            return max(0.0, m * 60 + s)
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            return max(0.0, h * 3600 + m * 60 + s)
    except (ValueError, IndexError):
        pass
    return 0.0


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _chunk_bounds_union(chunks: list[dict]) -> tuple[float, float]:
    t0 = min(chunk_time_range(c)[0] for c in chunks)
    t1 = max(chunk_time_range(c)[1] for c in chunks)
    return float(t0), float(t1)


def format_context_blocks(chunks: list[dict]) -> str:
    """Build retrieval context with precise second ranges + text per chunk."""
    parts = []
    for c in chunks:
        a, b = chunk_time_range(c)
        parts.append(f"[{a:.3f}s – {b:.3f}s]\n{c['text'].strip()}")
    return "\n\n".join(parts)


def detect_mode(query: str) -> str:
    q = (query or "").lower()
    if any(
        x in q
        for x in (
            "not able to understand",
            "don't understand",
            "confused",
            "simplify",
            "simple terms",
            "eli5",
        )
    ):
        return "simple"
    if any(
        x in q
        for x in ("in detail", "explain in detail", "elaborate", "deep dive", "thoroughly")
    ):
        return "detailed"
    if "analogy" in q or "like what" in q:
        return "analogy"
    if "example" in q or "for instance" in q:
        return "example"
    return "default"


def _meaningful_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{4,}", text.lower()))


def grounding_score(answer: str, transcript: str) -> float:
    """Share of distinct 4+ char tokens from the answer that appear in the transcript."""
    a = _meaningful_tokens(answer)
    if not a:
        return 1.0
    tn = transcript.lower()
    hits = sum(1 for w in a if w in tn)
    return hits / len(a)


def _extraction_is_empty(extraction: str) -> bool:
    x = (extraction or "").strip().upper()
    if not x:
        return True
    if "NOT_IN_TRANSCRIPT" in x or "NOTHING RELEVANT" in x:
        return True
    return False


TEACHER_SYSTEM = """You are an AI teacher.
Answer ONLY using the provided transcript.
STRICT RULES:

* Do NOT add any external knowledge
* Do NOT generalize or summarize aggressively
* Preserve all important details
* If something is not present, say 'Not mentioned in the video'
* Always include the exact timestamp range in your JSON fields (timestamp_start, timestamp_end) — use the numbers given in the prompt, do not invent times."""


def _run_extraction(
    *,
    query: str,
    transcript_block: str,
    model: str,
) -> str:
    provider = get_provider(model=model)
    system = (
        "You extract information from a video transcript excerpt. "
        "Do NOT summarize away details. Do NOT paraphrase into fewer technical facts. "
        "List every important point, term, number, and step that appears, as separate bullets. "
        "If the excerpt has nothing relevant to the question, reply with exactly: NOT_IN_TRANSCRIPT"
    )
    user = f"""TRANSCRIPT EXCERPT:
{transcript_block}

QUESTION:
{query.strip()}

Extract ALL important points from this transcript that relate to the question. Do NOT summarize. Do NOT remove details."""
    return provider.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    ).strip()


def _run_explanation_json(
    *,
    query: str,
    transcript_block: str,
    extraction: str,
    timestamp_start: float,
    timestamp_end: float,
    mode: str,
    model: str,
    span_description: str = "retrieval window",
) -> str:
    provider = get_provider(model=model)
    mode = mode if mode in _MODE_HINTS else "default"
    style = (
        _MODE_HINTS[mode]
        if mode != "default"
        else "Be clear, structured, and encouraging like a good teacher."
    )
    user = f"""TRANSCRIPT (ground truth — {span_description}, seconds {timestamp_start:.3f} to {timestamp_end:.3f}):
{transcript_block}

EXTRACTED POINTS (do not contradict; use as a checklist of details to explain):
{extraction}

QUESTION:
{query.strip()}

TASK — STEP 2 EXPLANATION:
Explain these points clearly like a teacher. Keep all technical details intact. {style}

You MUST output ONLY valid JSON (no markdown fences) with this exact shape:
{{
  "answer": "<detailed explanation grounded only in the transcript>",
  "timestamp_start": {timestamp_start},
  "timestamp_end": {timestamp_end},
  "key_points": ["<string>", "..."]
}}

Rules:
- timestamp_start and timestamp_end MUST be exactly {timestamp_start} and {timestamp_end} (floats).
- key_points: 3–8 short strings, each traceable to the transcript.
- If the transcript does not support an answer, set "answer" to "{NOT_MENTIONED}" and key_points to []."""
    return provider.chat(
        model=model,
        messages=[
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user", "content": user},
        ],
    ).strip()


def _parse_teaching_json(raw: str) -> dict[str, Any] | None:
    text = _strip_json_fences(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Teaching JSON parse failed for chat response")
        return None
    if not isinstance(data, dict):
        return None
    return data


def _validate_and_finalize(
    data: dict[str, Any],
    *,
    transcript: str,
    ts0: float,
    ts1: float,
    extraction: str,
) -> dict[str, Any]:
    """Clamp times, check grounding, fill defaults."""
    answer = str(data.get("answer") or "").strip()
    key_points = data.get("key_points") or []
    if not isinstance(key_points, list):
        key_points = []
    key_points = [str(x).strip() for x in key_points if str(x).strip()]

    try:
        t0 = float(data.get("timestamp_start", ts0))
        t1 = float(data.get("timestamp_end", ts1))
    except (TypeError, ValueError):
        t0, t1 = ts0, ts1

    t0 = max(0.0, min(t0, ts1))
    t1 = max(t0, t1)

    combined = transcript + "\n" + extraction
    g = grounding_score(answer, combined)
    low = g < 0.38 and len(_meaningful_tokens(answer)) >= 6

    if answer.upper() == NOT_MENTIONED or NOT_MENTIONED in answer:
        pass
    elif low:
        answer = LOW_CONFIDENCE
        key_points = []

    return {
        "answer": answer,
        "timestamp_start": float(t0),
        "timestamp_end": float(t1),
        "key_points": key_points,
        "grounded": not low and answer != LOW_CONFIDENCE,
        "grounding_score": round(g, 4),
    }


def extract_timestamp_spans(text: str) -> list[dict]:
    """Parse 📍 mm:ss – mm:ss (or -) from model output; include seconds for video seek."""
    pat = re.compile(r"📍\s*(\d{1,2}:\d{2}(?:\.\d+)?)\s*[–-]\s*(\d{1,2}:\d{2}(?:\.\d+)?)")
    out = []
    for m in pat.finditer(text):
        a, b = m.group(1), m.group(2)
        out.append(
            {
                "start_label": a,
                "end_label": b,
                "start_sec": mmss_label_to_seconds(a),
                "end_sec": mmss_label_to_seconds(b),
            }
        )
    return out


def _resolve_sentence_timestamps(
    query: str,
    retrieved_chunks: list[dict],
    session_id: str | None,
    *,
    model: str,
) -> tuple[float, float, str]:
    """Prefer sentence-level start/end inside retrieved chunks; fallback to chunk union."""
    chunk_t0, chunk_t1 = _chunk_bounds_union(retrieved_chunks)
    if not session_id:
        return float(chunk_t0), float(chunk_t1), "chunk-level retrieval window"

    sentences = load_session_sentences(session_id)
    candidates = filter_sentences_overlapping_chunks(sentences, retrieved_chunks)
    best = find_most_relevant_sentence(query, candidates, embed_model=None)
    if best is None:
        return float(chunk_t0), float(chunk_t1), "chunk-level retrieval window"
    try:
        s0 = float(best["start"])
        s1 = float(best["end"])
    except (KeyError, TypeError, ValueError):
        return float(chunk_t0), float(chunk_t1), "chunk-level retrieval window"
    if s1 <= s0:
        return float(chunk_t0), float(chunk_t1), "chunk-level retrieval window"
    return s0, s1, "sentence-level (most relevant spoken sentence in retrieved chunks)"


def chat(
    query: str,
    retrieved_chunks: list[dict],
    *,
    model: str,
    mode: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Retrieve already done; multi-stage LLM; structured JSON + grounding.

    When ``session_id`` is set, ``transcript.json`` sentences refine ``timestamp_start`` /
    ``timestamp_end`` to the best-matching sentence inside retrieved chunks (FAISS unchanged).
    """
    model = normalize_llm_model_id(model)
    m = mode if mode is not None else detect_mode(query)

    if not retrieved_chunks:
        empty = {
            "answer": NOT_IN_VIDEO,
            "timestamp_start": 0.0,
            "timestamp_end": 0.0,
            "key_points": [],
            "timestamps": [],
            "mode": m,
            "grounded": True,
            "grounding_score": 1.0,
        }
        return empty

    transcript_block = format_context_blocks(retrieved_chunks)
    raw_transcript_only = "\n".join((c.get("text") or "").strip() for c in retrieved_chunks)
    ts0, ts1, span_desc = _resolve_sentence_timestamps(
        query, retrieved_chunks, session_id, model=model
    )

    extraction = _run_extraction(
        query=query,
        transcript_block=transcript_block,
        model=model,
    )
    if _extraction_is_empty(extraction):
        return {
            "answer": NOT_IN_VIDEO,
            "timestamp_start": float(ts0),
            "timestamp_end": float(ts1),
            "key_points": [],
            "timestamps": [
                {
                    "start_label": _mmss(ts0),
                    "end_label": _mmss(ts1),
                    "start_sec": float(ts0),
                    "end_sec": float(ts1),
                }
            ],
            "mode": m,
            "grounded": True,
            "grounding_score": 1.0,
        }

    raw2 = _run_explanation_json(
        query=query,
        transcript_block=transcript_block,
        extraction=extraction,
        timestamp_start=ts0,
        timestamp_end=ts1,
        mode=m,
        model=model,
        span_description=span_desc,
    )
    parsed = _parse_teaching_json(raw2)
    if not parsed:
        fallback = (
            f"📍 {_mmss(ts0)} – {_mmss(ts1)}\n\n"
            f"{extraction}\n\n"
            "(The model did not return valid JSON; showing extraction only.)"
        )
        return {
            "answer": fallback,
            "timestamp_start": float(ts0),
            "timestamp_end": float(ts1),
            "key_points": [],
            "timestamps": [
                {
                    "start_label": _mmss(ts0),
                    "end_label": _mmss(ts1),
                    "start_sec": float(ts0),
                    "end_sec": float(ts1),
                }
            ],
            "mode": m,
            "grounded": False,
            "grounding_score": 0.0,
        }

    finalized = _validate_and_finalize(
        parsed,
        transcript=raw_transcript_only,
        ts0=ts0,
        ts1=ts1,
        extraction=extraction,
    )

    display = finalized["answer"]

    ts_list = [
        {
            "start_label": _mmss(finalized["timestamp_start"]),
            "end_label": _mmss(finalized["timestamp_end"]),
            "start_sec": float(finalized["timestamp_start"]),
            "end_sec": float(finalized["timestamp_end"]),
        }
    ]
    return {
        "answer": display,
        "timestamp_start": finalized["timestamp_start"],
        "timestamp_end": finalized["timestamp_end"],
        "key_points": finalized["key_points"],
        "timestamps": ts_list,
        "mode": m,
        "grounded": finalized["grounded"],
        "grounding_score": finalized["grounding_score"],
    }
