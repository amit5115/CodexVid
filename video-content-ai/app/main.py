"""CodexVid AI — FastAPI entry: CodexVid API + static learn UI + health."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app import __version__
from app.api.codexvid import router as codexvid_router
from app.api.health import router as health_router
from app.config import HOST, PORT, RELOAD, STATIC_DIR, ensure_dirs

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id
        start = time.perf_counter()
        method, path = request.method, request.url.path
        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("[%s] %s %s -> 500 (%.1fms)", request_id, method, path, elapsed, exc_info=True)
            raise
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("[%s] %s %s -> %d (%.1fms)", request_id, method, path, response.status_code, elapsed)
        response.headers["X-Request-ID"] = request_id
        return response


def create_app() -> FastAPI:
    ensure_dirs()

    application = FastAPI(
        title="CodexVid AI",
        description="Video → transcript → lesson + grounded chat (FAISS + faster-whisper)",
        version=__version__,
    )
    application.add_middleware(RequestLoggingMiddleware)

    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error("[%s] Unhandled exception on %s %s: %s", request_id, request.method, request.url.path, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc), "request_id": request_id},
        )

    application.include_router(health_router)
    application.include_router(codexvid_router)

    from fastapi.staticfiles import StaticFiles

    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    _learn_html = (STATIC_DIR / "learn.html").read_text(encoding="utf-8")
    _learn_headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}

    @application.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(_learn_html, headers=_learn_headers)

    @application.get("/learn", response_class=HTMLResponse)
    async def learn():
        return HTMLResponse(_learn_html, headers=_learn_headers)

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=RELOAD,
        reload_excludes=[".*", ".venv", "venv", "__pycache__", "data", "*.pyc"],
    )
