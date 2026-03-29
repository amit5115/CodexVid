"""Video acquisition: download from YouTube/URL and normalize media sources."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import yt_dlp

# Optional cookies.txt at the repo root for login-required videos.
_COOKIES_FILE = Path(__file__).resolve().parent.parent.parent.parent / "cookies.txt"

_YTDLP_SOCKET_TIMEOUT = float(os.getenv("VCAI_YTDLP_SOCKET_TIMEOUT", "120"))


def _find_node() -> str | None:
    """Return the absolute path to the node binary, or None if not found."""
    import shutil
    for candidate in (
        shutil.which("node"),
        "/opt/homebrew/bin/node",
        "/usr/local/bin/node",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _ydl_base_opts(*, use_cookies: bool = True) -> dict:
    """Common yt-dlp options: timeout, JS runtime, player clients, optional cookies."""
    node_path = _find_node()
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": _YTDLP_SOCKET_TIMEOUT,
        "js_runtimes": {"node": {"path": node_path}} if node_path else {},
        # Android/iOS clients don't require a JS runtime — most reliable.
        "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        },
    }
    if use_cookies:
        if _COOKIES_FILE.exists():
            opts["cookiefile"] = str(_COOKIES_FILE)
        else:
            for _br in ("chrome", "safari", "firefox", "edge", "brave"):
                try:
                    _test = {**opts, "cookiesfrombrowser": (_br,)}
                    with yt_dlp.YoutubeDL(_test) as _ydl:
                        _ydl.cookiejar
                    opts["cookiesfrombrowser"] = (_br,)
                    break
                except Exception:
                    continue
            else:
                opts["cookiesfrombrowser"] = ("chrome",)
    return opts


def is_url(source: str) -> bool:
    """Return True if *source* starts with http:// or https://."""
    return bool(re.match(r"https?://", source))


def normalize_media_source(source: str) -> str:
    """Prepend https:// to bare YouTube/video host strings that are missing a scheme.

    Browsers and mobile apps often copy ``youtube.com/...`` without the scheme;
    this makes sure ``is_url()`` returns True for those strings.
    """
    s = (source or "").strip()
    if not s or is_url(s):
        return s
    try:
        p = Path(s).expanduser()
        if p.exists() and p.is_file():
            return s
    except OSError:
        pass
    lowered = s.lower()
    host_hints = (
        "youtube.com", "youtu.be", "m.youtube.com",
        "vimeo.com", "tiktok.com", "instagram.com",
        "facebook.com", "fb.watch",
        "twitter.com", "x.com",
        "sharepoint.com", "1drv.ms", "onedrive.",
    )
    if lowered.startswith("www.") or any(h in lowered for h in host_hints):
        return "https://" + s.lstrip("/")
    return s


def download_video(url: str, output_dir: Path) -> Path:
    """Download *url* to *output_dir* and return the local video file path.

    Tries multiple format strings and both cookie/no-cookie modes for resilience.
    """
    output_template = str(output_dir / "%(title)s.%(ext)s")
    formats = [
        "18/22/bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=480]+bestaudio/best[ext=mp4]/best",
        "bv*+ba/bestvideo+bestaudio/b",
        "best",
    ]

    last_err = None
    for fmt in formats:
        for use_cookies in (True, False):
            ydl_opts = {
                **_ydl_base_opts(use_cookies=use_cookies),
                "format": fmt,
                "outtmpl": output_template,
                "merge_output_format": "mp4",
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)
                last_err = None
                break
            except yt_dlp.utils.DownloadError as e:
                last_err = e
                continue
        if last_err is None:
            break

    if last_err is not None:
        raise last_err

    video_path = Path(filename)
    if not video_path.exists():
        mp4_path = video_path.with_suffix(".mp4")
        if mp4_path.exists():
            video_path = mp4_path

    return video_path
