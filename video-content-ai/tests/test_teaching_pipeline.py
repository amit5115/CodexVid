"""Unit tests for per-chunk teaching merge/coverage (no LLM)."""

from app.codexvid.teaching import enforce_coverage, merge_adjacent_topics


def test_merge_adjacent_only_when_similar_and_consecutive():
    topics = [
        {"topic_title": "Intro to Python", "description": "a", "start_time": 0.0, "end_time": 30.0},
        {"topic_title": "Intro to Python", "description": "b", "start_time": 30.0, "end_time": 60.0},
        {"topic_title": "Something else", "description": "c", "start_time": 60.0, "end_time": 90.0},
    ]
    out = merge_adjacent_topics(topics, similarity_threshold=0.9)
    assert len(out) == 2
    assert out[0]["end_time"] == 60.0
    assert "Intro to Python" in out[0]["topic_title"]


def test_enforce_coverage_extends_bounds():
    chunks = [
        {"text": "a", "start_time": 5.0, "end_time": 10.0, "start": 5.0, "end": 10.0},
        {"text": "b", "start_time": 10.0, "end_time": 100.0, "start": 10.0, "end": 100.0},
    ]
    topics = [
        {"topic_title": "t1", "description": "", "start_time": 6.0, "end_time": 50.0},
        {"topic_title": "t2", "description": "", "start_time": 50.0, "end_time": 99.0},
    ]
    out = enforce_coverage(topics, chunks)
    assert out[0]["start_time"] == 5.0
    assert out[-1]["end_time"] == 100.0
