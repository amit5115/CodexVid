"""AWS Transcribe helpers: JSON parsing and mocked job flow."""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services import aws_transcribe
from app.services.aws_transcribe import parse_transcript_json_to_segments, transcribe_path_to_segments


def test_parse_transcript_json_to_segments_groups_words():
    data = {
        "results": {
            "items": [
                {
                    "id": "0",
                    "type": "pronunciation",
                    "start_time": "0.5",
                    "end_time": "0.9",
                    "alternatives": [{"content": "Hello"}],
                },
                {
                    "id": "1",
                    "type": "pronunciation",
                    "start_time": "1.0",
                    "end_time": "1.4",
                    "alternatives": [{"content": "world"}],
                },
                {
                    "id": "2",
                    "type": "punctuation",
                    "alternatives": [{"content": "."}],
                },
            ],
        }
    }
    segs = parse_transcript_json_to_segments(data, max_span_sec=30.0)
    assert len(segs) == 1
    assert segs[0]["start"] == 0.5
    assert segs[0]["end"] == 1.4
    assert "Hello" in segs[0]["text"] and "world" in segs[0]["text"]


def test_parse_transcript_json_empty_items():
    assert parse_transcript_json_to_segments({}) == []
    assert parse_transcript_json_to_segments({"results": {}}) == []


@pytest.fixture
def tiny_wav(tmp_path: Path) -> Path:
    p = tmp_path / "t.wav"
    p.write_bytes(b"dummy")
    return p


def test_transcribe_path_to_segments_mocked(tiny_wav: Path):
    transcript_json = {
        "results": {
            "items": [
                {
                    "type": "pronunciation",
                    "start_time": "0.0",
                    "end_time": "0.5",
                    "alternatives": [{"content": "Hi"}],
                },
            ],
        },
    }

    transcript_uri = "https://example.invalid/transcript.json"

    mock_transcribe = MagicMock()
    mock_transcribe.start_transcription_job.return_value = {}
    mock_transcribe.get_transcription_job.return_value = {
        "TranscriptionJob": {
            "TranscriptionJobStatus": "COMPLETED",
            "Transcript": {"TranscriptFileUri": transcript_uri},
        },
    }

    mock_s3 = MagicMock()

    def fake_client(name: str, region_name: str | None = None):
        if name == "transcribe":
            return mock_transcribe
        if name == "s3":
            return mock_s3
        raise AssertionError(name)

    with patch.multiple(
        aws_transcribe,
        AWS_TRANSCRIBE_BUCKET="test-bucket",
        AWS_REGION="us-east-1",
    ):
        with patch("app.services.aws_transcribe.boto3.client", side_effect=fake_client):
            with patch(
                "app.services.aws_transcribe.urlopen",
                return_value=BytesIO(json.dumps(transcript_json).encode("utf-8")),
            ):
                segs = transcribe_path_to_segments(tiny_wav, "en")

    assert len(segs) == 1
    assert segs[0]["text"] == "Hi"
    mock_s3.upload_file.assert_called_once()
    mock_transcribe.start_transcription_job.assert_called_once()
    mock_s3.delete_object.assert_called_once()
