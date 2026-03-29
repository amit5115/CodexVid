"""Unit tests for transcript merge/align and chapter clean_timestamps."""

from app.codexvid.timestamp_utils import (
    clean_timestamps,
    dedupe_overlapping_words,
    merge_segments,
    normalize_transcript_segments,
    words_to_fine_segments,
)


def test_merge_segments_resolves_overlap():
    segs = [
        {"text": "hello world", "start": 0.0, "end": 2.0},
        {"text": "hello world", "start": 0.5, "end": 2.5},
    ]
    out = merge_segments(segs)
    assert len(out) == 1
    assert out[0]["end"] >= out[0]["start"]


def test_dedupe_overlapping_words():
    words = [
        {"word": "hi", "start": 1.0, "end": 1.1},
        {"word": "hi", "start": 1.05, "end": 1.15},
        {"word": "there", "start": 1.2, "end": 1.5},
    ]
    out = dedupe_overlapping_words(words)
    assert len(out) == 2


def test_words_to_fine_segments():
    words = []
    t = 0.0
    for _ in range(20):
        words.append({"word": "w", "start": t, "end": t + 0.2})
        t += 0.2
    segs = words_to_fine_segments(words, min_sec=2.0, max_sec=5.0)
    assert len(segs) >= 1
    for s in segs:
        assert s["end"] > s["start"]


def test_normalize_transcript_segments():
    raw = [
        {"text": "a", "start": 1.0, "end": 1.5},
        {"text": "b", "start": 1.4, "end": 2.0},
    ]
    out = normalize_transcript_segments(raw)
    assert out[1]["start"] >= out[0]["end"] - 0.15


def test_clean_timestamps_chapters():
    chapters = [
        {"title": "A", "start": 10, "end": 50},
        {"title": "B", "start": 45, "end": 80},
    ]
    out = clean_timestamps(chapters)
    assert len(out) >= 1
    for i in range(len(out) - 1):
        assert out[i + 1]["start"] >= out[i]["end"]
