"""Video acquisition and media processing.

Handles downloading, audio extraction, clip cutting, and frame extraction.
All heavy ffmpeg/yt-dlp work lives here.
"""

# =============================================================================
# BEGINNER HEADER — what this file is (read this first)
# =============================================================================
# Think of this file as the “video workshop” for the app. Other parts of the
# program (web API routes, command-line tools) send you a YouTube link or a
# file path; this workshop downloads the movie if needed, slices it, pulls out
# sound, grabs still pictures, and can turn wide videos into tall phone clips.
#
# Main outside callers (who knocks on this door):
#   • app/api/media.py      — thumbnails, clip generation, shorts generation
#   • app/api/transcription.py — turns video into audio before speech-to-text
#   • app/cli.py          — same idea from the terminal
#   • app/api/batch.py    — lists all videos in a YouTube playlist
#   • app/api/history.py  — fetches title/channel/thumbnail without downloading
#
# Tools this file relies on (must be installed on the computer like apps):
#   • yt-dlp — downloads from YouTube and reads playlist metadata (like a
#     specialized browser that saves video files).
#   • ffmpeg / ffprobe — converts video, cuts clips, extracts audio and frames
#     (like a Swiss Army knife for video files).
#
# Names starting with _ (underscore) are “internal helpers”: only other
# functions inside this same file are meant to use them—not the rest of the app.
# =============================================================================

# “from __future__ import annotations” tells Python 3.7+ to allow type hints
# (like Path, dict) to be written in a slightly friendlier way without breaking
# older rules. Nothing visible changes at runtime for you; it’s for the language.
from __future__ import annotations

# json — Python’s built-in way to read/write text that looks like JavaScript
# objects (used here to read ffprobe’s machine-readable “report” about a video).
import json

# os — environment variables (e.g. yt-dlp socket timeout override).
import os

# re — “regular expressions”: a mini language for finding patterns in text
# (e.g. “does this string start with http?” or “find timestamps in a paragraph”).
import re

# subprocess — lets Python run other programs (ffmpeg, ffprobe) as if you typed
# them in a terminal, and capture their output or errors.
import subprocess

# tempfile — creates empty scratch folders/files that the OS can clean up later;
# we put downloaded videos there so we don’t clutter the user’s Desktop.
import tempfile

# Path — a nicer way to handle file paths than plain strings (join folders,
# check “does this file exist?”, change file extension, etc.).
from pathlib import Path

# yt_dlp — Python library that wraps yt-dlp/YouTube downloading (the same family
# of tool as youtube-dl). No code in this file “calls” the import itself; the
# name yt_dlp is used when we build a YoutubeDL object later.
import yt_dlp

# Path to an optional cookies.txt next to the project root (four parents up from
# this file: services → app → video-content-ai → AI-K8S). If present, YouTube
# may treat downloads like a logged-in browser (helps when videos need sign-in).
# Only _ydl_base_opts and functions that use it read this constant.
_COOKIES_FILE = Path(__file__).resolve().parent.parent.parent.parent / "cookies.txt"  # optional Netscape cookies at repo root

# yt-dlp defaults can use ~20s read timeouts; googlevideo.com segments often need longer on slow links.
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
    # Who calls this: download_video, extract_playlist_urls, fetch_youtube_info (only from inside this file).
    """Common yt-dlp options with YouTube auth and JS challenge solving."""
    node_path = _find_node()
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        # Avoid HTTPSConnectionPool read timeout=20 against googlevideo CDN on slow networks.
        "socket_timeout": _YTDLP_SOCKET_TIMEOUT,
        # Provide explicit node path so yt-dlp finds it even when the server PATH is minimal.
        "js_runtimes": {"node": {"path": node_path}} if node_path else {},
        # android/ios clients don't need JS/PO-token — most reliable without a JS runtime.
        "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        },
    }
    # If the caller wants login-like behavior, try cookies; otherwise skip this block.
    if use_cookies:
        # If the user dropped a cookies.txt file in the repo root, use that file.
        if _COOKIES_FILE.exists():
            # Tell yt-dlp exactly which file holds browser cookies (string path).
            opts["cookiefile"] = str(_COOKIES_FILE)
        else:
            # No file? Ask yt-dlp to read cookies directly from Chrome on this machine.
            # Analogy: “borrow Chrome’s keyring” instead of a copied cookies file.
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
    # Who calls this: process_video, generate_clips_from_text, generate_shorts_from_text, app/api/media.py.
    # re.match checks the *start* of the string against the pattern.
    # https? means http or https; :// is literal. bool(...) turns “match or None” into True/False.
    # Called by: process_video, generate_clips_from_text, generate_shorts_from_text,
    # and app/api/media.py when deciding URL vs local file.
    return bool(re.match(r"https?://", source))


def normalize_media_source(source: str) -> str:
    """If the user pasted a video URL without http(s)://, prepend https:// so is_url() works.

    Browsers and mobile apps often copy ``youtube.com/...`` or ``www.youtube.com/...``
    without a scheme; we must not treat those as local filesystem paths.
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


def _is_sharepoint_url(url: str) -> bool:
    """Check if a URL points to SharePoint, OneDrive, or Teams recording storage."""
    lower = url.lower()
    return any(d in lower for d in (
        "sharepoint.com", "onedrive.live.com", "1drv.ms",
        "-my.sharepoint.com",
    ))


def _sharepoint_download_url(url: str) -> str:
    """Convert a SharePoint/OneDrive sharing link to a direct download URL."""
    if "sharepoint.com" in url.lower() and "download=1" not in url.lower():
        sep = "&" if "?" in url else "?"
        return url + sep + "download=1"
    return url


def _download_direct(url: str, output_dir: Path, filename_hint: str = "") -> Path:
    """Download a file via HTTP (for SharePoint, OneDrive, or any direct link)."""
    from urllib.parse import unquote, urlparse

    import httpx

    download_url = _sharepoint_download_url(url) if _is_sharepoint_url(url) else url

    try:
        with httpx.stream("GET", download_url, follow_redirects=True, timeout=300) as resp:
            resp.raise_for_status()

            ct_disp = resp.headers.get("content-disposition", "")
            fname = ""
            if "filename=" in ct_disp:
                match = re.search(r'filename[*]?=["\']?([^"\';\r\n]+)', ct_disp)
                fname = match.group(1).strip() if match else ""
            if not fname:
                path_part = unquote(urlparse(url).path)
                fname = Path(path_part).name if Path(path_part).suffix else ""
            if not fname or not Path(fname).suffix:
                fname = filename_hint or "meeting_recording.mp4"

            out_path = output_dir / fname
            with open(out_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
        return out_path
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            raise RuntimeError(
                "SharePoint/OneDrive link requires login. "
                "Please download the recording to your computer first, "
                "then upload it using the file upload area above."
            ) from exc
        raise


def download_video(url: str, output_dir: Path) -> Path:
    output_template = str(output_dir / "%(title)s.%(ext)s")
    formats = [
        # Single-file progressive formats first: no merge needed, works without ffmpeg/JS runtime.
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


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    # Who calls this: only process_video (inside this file).
    # Output WAV named like the video stem (filename without extension) in output_dir.
    audio_path = output_dir / f"{video_path.stem}.wav"
    # argv-style list: each item becomes one “word” on the ffmpeg command line (like typing in Terminal).
    cmd = [
        "ffmpeg",  # the converter program (must be installed and on PATH)
        "-i", str(video_path),  # -i = input file path (string so ffmpeg understands it)
        "-vn",  # video no — strip the picture; audio only (like recording the soundtrack)
        "-acodec", "pcm_s16le",  # PCM 16-bit little-endian: simple uncompressed WAV-friendly audio
        "-ar", "16000",  # sample rate 16 kHz: common for speech recognition models
        "-ac", "1",  # one audio channel = mono (smaller file; speech doesn’t need stereo)
        "-y",  # yes overwrite output if the wav already exists
        str(audio_path),  # output file path
    ]
    # capture_output=True hides ffmpeg’s chatter; check=True turns a failed run into an exception.
    subprocess.run(cmd, capture_output=True, check=True)
    return audio_path  # tell the caller exactly where the new .wav lives


def process_video(source: str) -> tuple[Path, Path]:
    # Who calls this: app/api/transcription.py, app/cli.py, app/api/media.py (transcribe / preview flow).
    """Download/locate video, extract audio.  Returns (audio_path, temp_dir)."""
    source = normalize_media_source(source)
    source = _canonical_youtube_download_source(source)
    # Create a new empty temp directory whose name starts with video-content-ai-.
    tmp_dir = Path(tempfile.mkdtemp(prefix="video-content-ai-"))

    # Branch: SharePoint/OneDrive link, YouTube/other URL, or local file.
    if is_url(source) and _is_sharepoint_url(source):
        video_path = _download_direct(source, tmp_dir)
    elif is_url(source):
        video_path = download_video(source, tmp_dir)
    else:
        # Treat source as a filesystem path (e.g. /Users/.../movie.mp4).
        video_path = Path(source)
        # If the file isn’t there, fail clearly (FileNotFoundError is standard Python).
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {source}")

    # Run ffmpeg to produce a .wav next to (or inside) tmp_dir for speech recognition.
    audio_path = extract_audio(video_path, tmp_dir)
    # Tuple: (where the wav is, where scratch files live). Callers delete tmp_dir when done.
    return audio_path, tmp_dir


def download_audio_only(url: str, output_dir: Path) -> Path:
    """Download only the audio stream from a URL — much faster than full video."""
    output_template = str(output_dir / "%(title)s.%(ext)s")
    audio_formats = [
        "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        "bestaudio/best",
    ]

    last_err = None
    for fmt in audio_formats:
        for use_cookies in (True, False):
            ydl_opts = {
                **_ydl_base_opts(use_cookies=use_cookies),
                "format": fmt,
                "outtmpl": output_template,
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
        return download_video(url, output_dir)

    audio_path = Path(filename)
    if not audio_path.exists():
        for ext in (".m4a", ".webm", ".opus", ".mp3", ".ogg"):
            alt = audio_path.with_suffix(ext)
            if alt.exists():
                return alt
        return download_video(url, output_dir)

    return audio_path


def process_video_fast(source: str) -> tuple[Path, Path]:
    """Fast audio extraction: audio-only download for URLs, direct extract for files.

    Returns (audio_wav_path, temp_dir) — same contract as process_video but skips
    full video download for remote URLs.
    """
    source = normalize_media_source(source)
    source = _canonical_youtube_download_source(source)
    tmp_dir = Path(tempfile.mkdtemp(prefix="video-content-ai-"))

    if is_url(source) and _is_sharepoint_url(source):
        video_path = _download_direct(source, tmp_dir)
        audio_path = extract_audio(video_path, tmp_dir)
    elif is_url(source):
        audio_file = download_audio_only(source, tmp_dir)
        audio_path = extract_audio(audio_file, tmp_dir)
    else:
        video_path = Path(source)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {source}")
        audio_path = extract_audio(video_path, tmp_dir)

    return audio_path, tmp_dir


# ── Clip cutting ────────────────────────────────────────────────────
# Visual divider only: everything below cuts a time range out of a video file.

def cut_clip(video_path: Path, start_sec: float, end_sec: float, output_path: Path) -> Path:
    # Who calls this: generate_clips_from_text, generate_shorts_from_text (HTTP API uses those).
    cmd = [
        "ffmpeg",  # video editor CLI
        "-y",  # overwrite destination if present
        "-i", str(video_path),  # full source movie
        "-ss", str(start_sec),  # seek to this second (start of clip)
        "-to", str(end_sec),  # end time of clip (duration is end minus start)
        "-c:v", "libx264",  # re-encode video with H.264 (widely supported)
        "-c:a", "aac",  # re-encode audio as AAC
        "-preset", "fast",  # encoding speed vs compression tradeoff
        "-movflags", "+faststart",  # metadata at front — faster start when streamed over the web
        str(output_path),  # where to write the new shorter mp4
    ]
    subprocess.run(cmd, capture_output=True, check=True)  # run; raise if ffmpeg exits with error
    return output_path  # same Path object the caller passed (now points to a real file)


def parse_timestamp_to_seconds(ts: str) -> float:
    # Who calls this: generate_clips_from_text, generate_shorts_from_text only (inside this file).
    # Strip whitespace, split on ":", convert each piece to float (e.g. "1:05" → [1.0, 5.0]).
    parts = [float(p) for p in ts.strip().split(":")]
    # Three parts = hours:minutes:seconds (like 1:23:45).
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]  # h→sec, m→sec, add s
    # Two parts = minutes:seconds.
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]  # minutes to seconds plus seconds
    # One part = seconds only.
    return parts[0]  # already in seconds


def generate_clips_from_text(
    highlights_text: str, video_source: str, output_dir: Path,
) -> list[dict]:
    # Who calls this: app/api/media.py (clip-generation HTTP endpoint runs this in a worker thread).
    # Compiled regex once (faster if many matches). Finds "MM:SS - MM:SS" or with hours, unicode dashes, "to".
    timestamp_pattern = re.compile(
        r'(\d{1,2}:\d{2}(?::\d{2})?)\s*[-\u2013\u2014to]+\s*(\d{1,2}:\d{2}(?::\d{2})?)'  # group1=start time, group2=end time
    )
    # findall returns list of (start, end) string pairs.
    matches = timestamp_pattern.findall(highlights_text)
    # No timestamps → nothing to cut; return empty list to the API.
    if not matches:
        return []  # caller (media API) will report “no clips”

    # If video_source is a URL, we need a local file before ffmpeg can cut it.
    if is_url(video_source):
        tmp = Path(tempfile.mkdtemp(prefix="vcai-clip-"))
        video_path = download_video(video_source, tmp)
    else:
        # Already a path string → Path object for ffmpeg -i.
        video_path = Path(video_source)

    # Split full text into lines so we can guess a human-readable title from the same line as a timestamp.
    lines = highlights_text.split("\n")
    # Will collect dicts describing each successful clip (title, times, filename).
    clips: list[dict] = []

    # For each start/end pair found by the regex in the whole blob of text.
    for start_ts, end_ts in matches:
        # Convert "1:30" style strings to seconds as numbers for ffmpeg.
        start_sec = parse_timestamp_to_seconds(start_ts)
        end_sec = parse_timestamp_to_seconds(end_ts)
        # Skip nonsensical ranges (end before start).
        if end_sec <= start_sec:
            continue  # ignore this pair and move to the next regex match

        # Default filename prefix if we can’t infer a nicer title from text.
        title = f"clip_{len(clips) + 1}"
        # Look for the line that contains this exact start timestamp string.
        for line in lines:
            if start_ts in line:
                # Remove the timestamp range from the line to leave a description.
                cleaned = re.sub(r'[\d:]+\s*[-\u2013\u2014to]+\s*[\d:]+', '', line)
                # Strip leading bullets, numbers, markdown junk.
                cleaned = re.sub(r'^[-*#\d.\s]+', '', cleaned).strip()
                # Remove parenthetical notes like (intro).
                cleaned = re.sub(r'\s*\(.*?\)\s*', '', cleaned).strip()
                if cleaned:
                    # Sanitize to letters/digits/spaces/hyphen; max 50 chars for safe filenames.
                    title = re.sub(r'[^\w\s-]', '', cleaned)[:50].strip()  # drop weird punctuation for OS safety
                break  # stop scanning lines once we handled the line containing this start_ts

        # Build output path: spaces → underscores, .mp4 extension.
        clip_path = output_dir / f"{title.replace(' ', '_')}.mp4"
        try:
            cut_clip(video_path, start_sec, end_sec, clip_path)
            # Record metadata the UI or API can return as JSON.
            clips.append({
                "title": title,  # human label for UI or download name
                "start": start_ts,  # original timestamp text from highlights (unchanged)
                "end": end_ts,  # same for end
                "duration": round(end_sec - start_sec, 1),  # length in seconds, one decimal
                "filename": clip_path.name,  # basename only (no folders)
                "path": str(clip_path),  # full path string for servers that need absolute location
            })
        except subprocess.CalledProcessError:
            # ffmpeg failed for this segment; skip it and try the next match.
            continue

    return clips  # list of dicts — may be shorter than matches if some ffmpeg runs failed


# ── Frame extraction ────────────────────────────────────────────────
# Divider: still images (JPEGs) pulled from the video for thumbnails or vision AI.

def extract_frames(video_path: Path, output_dir: Path, interval: int = 30) -> list[tuple[Path, float]]:
    # Who calls this: app/api/media.py (thumbnail / frame strip endpoint).
    # Ensure folder exists (parents=True makes parent dirs too; exist_ok avoids error if there).
    output_dir.mkdir(parents=True, exist_ok=True)
    # ffprobe asks “how long is this file?” in JSON form instead of human text.
    cmd = [
        "ffprobe",  # probe tool (comes with ffmpeg)
        "-v", "quiet",  # less log noise
        "-print_format", "json",  # machine-readable output
        "-show_format", str(video_path),  # include container-level duration, etc.
    ]
    # text=True gives str stdout; check=True raises on ffprobe failure.
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Parse JSON string → Python dict, then read duration under "format".
    duration = float(json.loads(result.stdout)["format"]["duration"])

    # Pattern ffmpeg will use: frame_0001.jpg, frame_0002.jpg, ... (%04d = zero-padded width 4).
    pattern = str(output_dir / "frame_%04d.jpg")
    # fps=1/interval means one frame every `interval` seconds (e.g. every 30s).
    cmd = [
        "ffmpeg",  # frame exporter
        "-y",  # overwrite existing numbered jpgs if re-run
        "-i", str(video_path),  # input video
        "-vf", f"fps=1/{interval}",  # one still image every `interval` seconds
        "-q:v", "2",  # JPEG quality knob (lower number ≈ higher quality here)
        pattern,  # output filename pattern with %04d
    ]
    subprocess.run(cmd, capture_output=True, check=True)  # run ffmpeg; fail if extraction breaks

    # Build list of (path, timestamp_in_seconds) for frames that actually exist and aren’t empty.
    frames: list[tuple[Path, float]] = []  # will fill with (jpeg path, seconds into video)
    idx = 0  # 0-based index; ffmpeg filenames are 1-based (frame_0001.jpg)
    ts = 0.0  # timestamp we associate with the frame at this index
    while ts < duration:
        # ffmpeg names from 1: frame_0001.jpg corresponds to idx 0 at time 0.
        frame_path = output_dir / f"frame_{idx + 1:04d}.jpg"
        # st_size > 0 skips corrupt zero-byte files.
        if frame_path.exists() and frame_path.stat().st_size > 0:
            frames.append((frame_path, ts))  # remember file + when in the video it came from
        idx += 1  # next numbered frame file
        ts += interval  # next sample time (e.g. +30 seconds)

    return frames  # caller may upload these images or show a filmstrip UI


# ── Playlist ────────────────────────────────────────────────────────
# (Spacer section in source layout; playlist URL helper lives lower in this file.)

# ── Vertical shorts (9:16) ───────────────────────────────────────
# Phone-shaped video: crop wide desktop video to tall portrait for TikTok/Reels/Shorts.

def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    # Who calls this: create_vertical_short only (inside this file).
    """Return (width, height) of a video file using ffprobe."""
    # Local alias avoids shadowing the module-level json import name in this function’s scope (style/clarity).
    import json as _json
    # -show_streams asks about tracks; -select_streams v:0 picks the first video track only.
    cmd = [
        "ffprobe",  # inspect media file
        "-v", "quiet",  # minimal logging
        "-print_format", "json",  # structured output
        "-show_streams",  # include per-track info
        "-select_streams", "v:0",  # only first video track (ignore audio/subtitles)
        str(video_path),  # file to inspect
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # run probe
    # "streams" is a list; [0] is the first video stream’s metadata dict.
    streams = _json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")  # not a video or corrupt file
    # width/height come as numbers in JSON; int() for type consistency.
    return int(streams[0]["width"]), int(streams[0]["height"])  # tuple: pixels wide, pixels tall


def _get_video_duration(video_path: Path) -> float:
    # Internal helper (leading _): only generate_shorts_from_text uses this.
    import json as _json  # local alias (same as get_video_dimensions — keeps pattern consistent)
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(video_path),  # duration lives under format in JSON
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(_json.loads(result.stdout)["format"]["duration"])  # seconds as float


def create_vertical_short(
    video_path: Path,
    output_path: Path,
    crop_position: str = "center",
    output_width: int = 1080,
    output_height: int = 1920,
) -> Path:
    # Who calls this: generate_shorts_from_text only (after cutting a horizontal clip).
    """Convert a landscape video to vertical (9:16) for Shorts/Reels/TikTok."""
    # w,h = width and height in pixels of the source file.
    w, h = get_video_dimensions(video_path)
    # Target aspect ratio width/height (1080/1920 ≈ 0.5625 for 9:16 portrait).
    target_ratio = output_width / output_height

    # If video is wider than tall relative to target, crop left/right; else crop top/bottom.
    if w / h > target_ratio:
        crop_h = h  # use full height of source
        # Ideal crop width in pixels; int() truncates; then force even width (encoder-friendly).
        crop_w = int(h * target_ratio)
        crop_w -= crop_w % 2  # make even (some codecs prefer even dimensions)
        if crop_position == "left":
            crop_x = 0  # keep left edge of frame
        elif crop_position == "right":
            crop_x = w - crop_w  # align crop box to right edge
        else:
            crop_x = (w - crop_w) // 2  # center horizontally (integer division)
        crop_y = 0  # no vertical offset when we’re cropping sides
    else:
        crop_w = w  # use full width
        crop_h = int(w / target_ratio)
        crop_h -= crop_h % 2  # even height
        crop_x = 0
        crop_y = (h - crop_h) // 2  # center the crop vertically

    # vf = video filter chain: crop rectangle, then scale to exact output size.
    vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={output_width}:{output_height}"
    cmd = [
        "ffmpeg",  # encoder
        "-y",  # overwrite output
        "-i", str(video_path),  # source (usually a short horizontal clip)
        "-vf", vf,  # apply crop+scale in one pass
        "-c:v", "libx264",  # video codec
        "-c:a", "aac",  # audio codec (keeps sound if present)
        "-preset", "fast",  # speed/size tradeoff
        "-movflags", "+faststart",  # web-friendly mp4 layout
        str(output_path),  # tall mp4 path
    ]
    subprocess.run(cmd, capture_output=True, check=True)  # encode or raise
    return output_path  # path to finished vertical mp4


# Minimum length in seconds for a “short”; shorter clips get padded in time (see below).
MIN_SHORT_DURATION = 15  # read by generate_shorts_from_text when extending sub-15s segments


def generate_shorts_from_text(
    highlights_text: str,
    video_source: str,
    output_dir: Path,
    crop_position: str = "center",
) -> list[dict]:
    # Who calls this: app/api/media.py (shorts-generation endpoint).
    """Parse timestamps, cut clips, convert each to vertical 9:16."""
    # Same inline “start - end” pattern as generate_clips_from_text.
    inline_pattern = re.compile(
        r'(\d{1,2}:\d{2}(?::\d{2})?)\s*[-\u2013\u2014to]+\s*(\d{1,2}:\d{2}(?::\d{2})?)'  # same idea as clip timestamps
    )
    matches = inline_pattern.findall(highlights_text)  # list of (start_str, end_str)

    # Fallback: AI might label lines as **Start** 1:00 / **End** 1:45 instead of one line.
    if not matches:
        start_pattern = re.compile(
            r'\*{0,2}[Ss]tart\*{0,2}\s*(?:timestamp)?[:\s*]+(\d{1,2}:\d{2}(?::\d{2})?)'  # capture time after “Start”
        )
        end_pattern = re.compile(
            r'\*{0,2}[Ee]nd\*{0,2}\s*(?:timestamp)?[:\s*]+(\d{1,2}:\d{2}(?::\d{2})?)'  # capture time after “End”
        )
        starts = start_pattern.findall(highlights_text)  # ordered list of start strings
        ends = end_pattern.findall(highlights_text)  # ordered list of end strings
        # Pair i-th start with i-th end (assumes same count/order in the text).
        matches = list(zip(starts, ends))  # e.g. [("1:00","1:20"), ...]

    if not matches:
        return []  # nothing to do — API returns empty shorts list

    lines = highlights_text.split("\n")  # for title sniffing (same trick as clips)

    if is_url(video_source):
        tmp = Path(tempfile.mkdtemp(prefix="vcai-shorts-"))  # scratch folder for download
        video_path = download_video(video_source, tmp)
    else:
        video_path = Path(video_source)  # local path string → Path

    # Need full length to avoid extending a clip past the real end of the video.
    video_duration = _get_video_duration(video_path)
    shorts: list[dict] = []  # accumulates metadata for each vertical file we produce

    for start_ts, end_ts in matches:
        start_sec = parse_timestamp_to_seconds(start_ts)  # numeric start
        end_sec = parse_timestamp_to_seconds(end_ts)  # numeric end
        if end_sec <= start_sec:
            continue  # ignore this pair and move to the next regex match
        duration = end_sec - start_sec  # raw segment length before padding rules
        # Platform shorts are usually under ~60s; skip huge ranges as probably mistakes.
        if duration > 90:
            continue  # skip this match entirely

        # If clip is shorter than MIN_SHORT_DURATION, grow it symmetrically when possible.
        if duration < MIN_SHORT_DURATION:
            needed = MIN_SHORT_DURATION - duration  # how many seconds we must add
            half = needed / 2  # try to grow equally on both sides (like centering a photo mat)
            new_start = max(0, start_sec - half)  # don’t go before 0:00 of the file
            new_end = min(video_duration, end_sec + half)  # don’t go past the video’s end
            still_needed = MIN_SHORT_DURATION - (new_end - new_start)  # if we hit a wall, leftover gap
            if still_needed > 0:
                if new_start == 0:
                    new_end = min(video_duration, new_end + still_needed)  # extend only forward if stuck at start
                else:
                    new_start = max(0, new_start - still_needed)  # else extend backward
            start_sec, end_sec = new_start, new_end  # replace with padded range
            duration = end_sec - start_sec  # recompute length after padding

        clip_idx = len(shorts) + 1  # 1-based counter for filenames
        title = f"short_{clip_idx}"  # default if we don’t find a better label
        # Try to find a "Title: ..." line near the timestamp line for a nicer name.
        for i, line in enumerate(lines):
            if start_ts in line:
                # Look at a small window of lines before/after for a Title: field.
                for ctx in lines[max(0, i - 2):i + 6]:
                    title_match = re.search(r'[Tt]itle[:\s]+(.+)', ctx)
                    if title_match:
                        candidate = title_match.group(1).strip().strip('"\'*')
                        if candidate:
                            title = re.sub(r'[^\w\s-]', '', candidate)[:50].strip()
                            break
                else:
                    # for-else: no break in inner loop → fall back to cleaning the timestamp line.
                    cleaned = re.sub(r'[\d:]+\s*[-\u2013\u2014to]+\s*[\d:]+', '', line)
                    cleaned = re.sub(r'^[-*#\d.\s]+', '', cleaned).strip()
                    cleaned = re.sub(r'\s*\(.*?\)\s*', '', cleaned).strip()
                    if cleaned:
                        title = re.sub(r'[^\w\s-]', '', cleaned)[:50].strip()
                break

        # Prefix with index so two shorts with same title don’t overwrite each other.
        safe_title = f"{clip_idx}_{title.replace(' ', '_')}"  # unique, filesystem-safe stem
        clip_path = output_dir / f"{safe_title}_clip.mp4"  # temporary horizontal cut
        short_path = output_dir / f"{safe_title}_vertical.mp4"  # final portrait file we keep

        try:
            cut_clip(video_path, start_sec, end_sec, clip_path)  # slice source to clip_path
            create_vertical_short(clip_path, short_path, crop_position)  # reframe to 9:16
            # Delete the temporary horizontal slice; only keep the vertical file.
            clip_path.unlink(missing_ok=True)  # missing_ok=True: no error if already gone
            shorts.append({
                "title": title,  # display name
                "start": start_ts,  # original text timestamp
                "end": end_ts,
                "duration": round(duration, 1),  # may reflect padding
                "filename": short_path.name,  # basename of vertical mp4
                "path": str(short_path),  # full path for downloads/API
            })
        except subprocess.CalledProcessError:
            continue  # skip failed segment; try next timestamp pair

    return shorts  # list of dicts for the API response


# ── Playlist ────────────────────────────────────────────────────────
# Divider: functions that treat a YouTube playlist as a list of video links.

def extract_playlist_urls(playlist_url: str) -> list[dict]:
    # Who calls this: app/api/batch.py (batch processing runs this in a thread pool).
    # extract_flat: fast list of entries without downloading each video’s full metadata.
    ydl_opts = {
        **_ydl_base_opts(),  # cookies/quiet/node like other downloads
        "extract_flat": True,  # shallow entries — enough for URLs/titles without full probe
        "force_generic_extractor": False,  # let yt-dlp pick the right site handler
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # download=False: metadata only (titles, ids, durations if available).
        info = ydl.extract_info(playlist_url, download=False)  # big dict describing playlist
        return [
            {
                # Flat entries may expose direct url or only id; build watch URL if needed.
                "url": e.get("url") or f"https://www.youtube.com/watch?v={e.get('id', '')}",
                "title": e.get("title", "Unknown"),  # fallback label
                "duration": e.get("duration", 0),  # seconds; may be 0 if unknown in flat mode
            }
            # Loop over playlist entries; skip None slots (deleted/private placeholders).
            for e in (info.get("entries") or [])
            if e
        ]


# ── YouTube info ────────────────────────────────────────────────────
# Divider: lightweight “business card” fetch for a single video (no big download).


def canonical_youtube_info_url(url: str) -> str:
    """Normalize to ``watch?v=VIDEO_ID`` so ``&list=`` / ``&index=`` don't break metadata or thumbnails."""
    from urllib.parse import parse_qs, urlparse

    s = (url or "").strip()
    if not s:
        return s
    try:
        parsed = urlparse(s)
        host = (parsed.netloc or "").lower()
        if "youtu.be" in host:
            seg = (parsed.path or "").strip("/").split("/")[0]
            if seg:
                return f"https://www.youtube.com/watch?v={seg}"
        if "youtube.com" in host:
            # Shorts and /live/ use path segments; query may be only ?feature=share (no v=).
            segments = [p for p in (parsed.path or "").split("/") if p]
            if len(segments) >= 2:
                kind, vid = segments[0].lower(), segments[1]
                if kind in ("shorts", "live") and vid:
                    out = f"https://www.youtube.com/watch?v={vid}"
                    return out
            qs = parse_qs(parsed.query)
            v = (qs.get("v") or [None])[0]
            if v:
                return f"https://www.youtube.com/watch?v={v}"
    except Exception:
        pass
    return s


def _canonical_youtube_download_source(source: str) -> str:
    """After :func:`normalize_media_source`, reduce YouTube URLs to ``watch?v=ID`` so yt-dlp does not follow ``&list=`` / playlist context."""
    if not is_url(source):
        return source
    try:
        from urllib.parse import urlparse

        netloc = (urlparse(source).netloc or "").lower()
        if "youtube.com" not in netloc and "youtu.be" not in netloc:
            return source
        canonical = canonical_youtube_info_url(source)
        return canonical
    except Exception:
        return source


def _youtube_thumbnail_url(info: dict) -> str:
    """yt-dlp often leaves ``thumbnail`` empty but fills ``thumbnails`` (list of size variants)."""
    t = info.get("thumbnail")
    if t:
        return str(t)
    thumbs = info.get("thumbnails")
    if isinstance(thumbs, list):
        for entry in reversed(thumbs):
            if isinstance(entry, dict):
                u = entry.get("url")
                if u:
                    return str(u)
    return ""


def _youtube_oembed_metadata(watch_url: str) -> dict | None:
    """YouTube oEmbed (public JSON) — works for many videos when yt-dlp is blocked (bot / formats).

    Does not include duration or view_count; those stay 0.
    """
    import json as _json
    import urllib.error
    import urllib.parse
    import urllib.request

    q = urllib.parse.urlencode({"url": watch_url, "format": "json"})
    api = f"https://www.youtube.com/oembed?{q}"
    try:
        req = urllib.request.Request(
            api,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = _json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    title = data.get("title") or ""
    channel = data.get("author_name") or ""
    thumb = data.get("thumbnail_url") or ""
    if not title and not channel and not thumb:
        return None
    return {
        "title": str(title),
        "channel": str(channel),
        "duration": 0,
        "thumbnail": str(thumb),
        "view_count": 0,
    }


def fetch_youtube_info(url: str) -> dict:
    # Who calls this: app/api/history.py (via run_in_executor so the web server stays async).
    # Metadata-only: try a few player_client orders if the default chain fails (YouTube changes often).
    # Try without cookies first (fast); if YouTube blocks (bot / no formats), retry with cookies / browser cookies.
    client_chains = (
        ["android", "ios", "web"],
        ["web"],
        ["tv_embedded"],
    )
    target = canonical_youtube_info_url(url)
    last_err: Exception | None = None
    for use_cookies in (False, True):
        base = _ydl_base_opts(use_cookies=use_cookies)
        for clients in client_chains:
            opts = {
                **base,
                "skip_download": True,
                "extractor_args": {"youtube": {"player_client": clients}},
            }
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(target, download=False)
                thumb = _youtube_thumbnail_url(info)
                return {
                    "title": info.get("title", ""),
                    "channel": info.get("uploader", ""),
                    "duration": info.get("duration", 0),
                    "thumbnail": thumb,
                    "view_count": info.get("view_count", 0),
                }
            except Exception as e:
                last_err = e
                continue
    oembed = _youtube_oembed_metadata(target)
    if oembed is not None:
        return oembed
    assert last_err is not None
    raise last_err
