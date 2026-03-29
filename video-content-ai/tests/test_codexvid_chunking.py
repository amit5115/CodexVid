"""Unit tests for CodexVid chunking (no Whisper / FAISS required for these)."""

from app.codexvid.chunking import create_chunks


def test_create_chunks_empty():
    assert create_chunks([]) == []


def test_create_chunks_single_segment_small():
    segs = [{"text": "one two three four five", "start": 0.0, "end": 5.0}]
    out = create_chunks(segs, chunk_size=3, overlap=1)
    assert len(out) >= 1
    assert "one" in out[0]["text"]
    assert out[0]["start_time"] >= 0.0
    assert out[0]["end_time"] <= 5.0
    assert out[0]["start"] == out[0]["start_time"]


def test_create_chunks_preserves_time_order():
    segs = [
        {"text": "a " * 100, "start": 0.0, "end": 10.0},
        {"text": "b " * 100, "start": 10.0, "end": 20.0},
    ]
    out = create_chunks(segs, chunk_size=50, overlap=10)
    for c in out:
        assert c["start_time"] < c["end_time"]


def test_detect_mode_from_chat_module():
    from app.codexvid.chat import detect_mode

    assert detect_mode("please simplify this") == "simple"
    assert detect_mode("explain in detail the architecture") == "detailed"
    assert detect_mode("give me an analogy") == "analogy"
    assert detect_mode("show an example") == "example"


def test_mmss_label_to_seconds():
    from app.codexvid.chat import mmss_label_to_seconds

    assert mmss_label_to_seconds("00:00") == 0
    assert mmss_label_to_seconds("01:30") == 90
    assert mmss_label_to_seconds("1:05:00") == 3900


def test_extract_timestamp_spans_parses_seconds():
    from app.codexvid.chat import extract_timestamp_spans

    text = "Here is the idea.\n📍 01:00 – 02:30\nMore text.\n📍 00:10 - 00:20"
    spans = extract_timestamp_spans(text)
    assert len(spans) == 2
    assert spans[0]["start_label"] == "01:00"
    assert spans[0]["start_sec"] == 60
    assert spans[0]["end_sec"] == 150
    assert spans[1]["start_sec"] == 10
    assert spans[1]["end_sec"] == 20
