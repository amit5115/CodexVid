"""CodexVid upload API: file vs YouTube URL validation (heavy pipeline mocked)."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    with patch("app.main.ensure_dirs"):
        app = create_app()
        with TestClient(app) as c:
            yield c


def test_codexvid_upload_requires_file_or_youtube(client):
    resp = client.post(
        "/api/codexvid/upload",
        data={"model": "llama3", "whisper_model": "base", "language": "en"},
    )
    assert resp.status_code == 400
    err = resp.json().get("error", "")
    assert "youtube" in err.lower() and "upload" in err.lower()


def test_codexvid_upload_rejects_non_youtube_url(client):
    resp = client.post(
        "/api/codexvid/upload",
        data={
            "youtube_url": "https://example.com/watch?v=abc",
            "model": "llama3",
            "whisper_model": "base",
            "language": "en",
        },
    )
    assert resp.status_code == 400
    assert "youtube" in resp.json().get("error", "").lower()


def test_codexvid_upload_youtube_uses_download_and_process(client, monkeypatch):
    def fake_download(url: str, tmp_dir: Path) -> Path:
        p = Path(tmp_dir) / "v.mp4"
        p.write_bytes(b"x")
        return p

    def fake_process_upload(
        video_path: Path,
        *,
        whisper_model: str,
        language: str,
        llm_model: str,
    ):
        return "sess_yt", {
            "session_id": "sess_yt",
            "teaching": {"chapters": []},
        }

    monkeypatch.setattr("app.api.codexvid.download_video", fake_download)
    monkeypatch.setattr("app.api.codexvid.process_upload", fake_process_upload)

    resp = client.post(
        "/api/codexvid/upload",
        data={
            "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "model": "llama3",
            "whisper_model": "base",
            "language": "en",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess_yt"
    assert data.get("source") == "youtube"
    assert "youtube.com" in data.get("youtube_url", "")
