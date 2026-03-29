"""Liveness and readiness probes for CodexVid AI backend."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok", "version": __version__, "product": "codexvid-ai"}


@router.get("/ready")
async def ready():
    """LLM provider reachable (Ollama/OpenAI/etc.); no database in this build."""
    checks: dict[str, str] = {}
    try:
        from app.core.llm import get_provider

        provider = get_provider()
        provider.list_models()
        checks["llm_provider"] = "ok"
    except Exception as e:
        checks["llm_provider"] = f"unavailable: {e}"

    ok = checks.get("llm_provider") == "ok"
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ready" if ok else "not_ready",
            "version": __version__,
            "checks": checks,
        },
    )
