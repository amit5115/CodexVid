"""Minimal CLI: serve CodexVid AI."""

from __future__ import annotations

import typer

from app.config import HOST, PORT, RELOAD

cli = typer.Typer(no_args_is_help=True)


@cli.command("serve")
def serve(
    host: str = typer.Option(HOST, "--host", "-h"),
    port: int = typer.Option(PORT, "--port", "-p"),
    reload: bool = typer.Option(RELOAD, "--reload/--no-reload"),
):
    """Run the FastAPI app with Uvicorn."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        reload_excludes=[".*", ".venv", "venv", "__pycache__", "data", "*.pyc"],
    )


if __name__ == "__main__":
    cli()
