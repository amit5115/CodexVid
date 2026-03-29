"""Minimal API tests: health + learn UI + workspace routes removed."""

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


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_ready_with_mock_llm(client):
    with patch("app.core.llm.get_provider") as gp:
        gp.return_value.list_models.return_value = ["llama3"]
        resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json().get("checks", {}).get("llm_provider") == "ok"


def test_index_serves_learn_ui(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "CodexVid" in resp.text
    assert "/static/learn.js" in resp.text


def test_learn_alias(client):
    resp = client.get("/learn")
    assert resp.status_code == 200
    assert "CodexVid" in resp.text


def test_legacy_workspace_route_gone(client):
    resp = client.post("/api/workspace/generate", json={"job_id": "x", "type": "summary", "model": "llama3"})
    assert resp.status_code == 404
