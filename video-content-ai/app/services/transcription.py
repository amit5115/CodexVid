# =============================================================================
# WHAT THIS FILE IS (read this first if you are new to programming)
# =============================================================================
# Think of this file as a "listening typist" for your app: you give it an audio
# file (like a voice memo or the sound from a video), and it writes down the
# words people said — similar to closed captions on TV, but stored as text
# the computer can search, summarize, or translate later.
#
# HOW IT FITS IN THE BIGGER APP
# • The main public entry point is the function `transcribe()` near the bottom.
#   WHO CALLS IT:
#   – The web API in `app/api/transcription.py` (function `_run_transcription_sync`)
#     when a user asks the server to transcribe a video or audio file.
#   – The same `_run_transcription_sync` is also invoked by batch and scheduled
#     jobs (`app/api/batch.py`, `app/api/scheduling.py`) — they are "indirect" callers.
#   – The command-line tool in `app/cli.py` when someone runs transcription
#     from the terminal.
# • The helper `_parse_language()` and the dictionary `LANGUAGE_NAMES` are also
#   imported by `app/api/transcription.py` to understand language choices from
#   the user (e.g. "auto", "Hindi", or "translate to English").
# • Everything whose name starts with `_` is a "private kitchen helper" — only
#   used inside this file, like prep cooks the diner never sees.
#
# KEY IDEAS IN PLAIN LANGUAGE
# • "Transcription" = turning speech into written text.
# • "Whisper" / Faster-Whisper = the AI model that does the listening (like a
#   very patient listener who has read a lot of languages).
# • "Chunking" = if the recording is long, we cut it into shorter pieces (like
#   reading a book chapter by chapter) so memory does not overflow and we can
#   report progress piece by piece.
# • "Diarization" (here, simplified) = guessing "Speaker A" vs "Speaker B"
#   from pauses — not perfect like a human, but useful; like guessing who is
#   talking when the phone line is fuzzy because one person pauses longer.
# =============================================================================

# Official one-line summary for Python's `help()` and documentation tools; like
# the subtitle on a book cover — short, for humans skimming the module list.
"""Speech-to-text transcription with chunking and heuristic diarization."""

# Lets us write type hints like `list[str]` on older Python versions; like a
# translator note so the type-checker understands modern spelling everywhere.
from __future__ import annotations

# `json` turns text in JSON format into Python data (and back); like unpacking
# a labeled box so we can read the "duration" label ffprobe prints.
import json

# `logging` writes messages to logs (files or console) for developers; like a
# diary the program keeps: "I loaded the model", "language was uncertain", etc.
import logging

# `subprocess` runs other programs (here: ffprobe, ffmpeg) as if you typed
# them in Terminal; like asking a specialist tool "how long is this file?"
import subprocess

# `tempfile` creates safe temporary folders/files that the OS can clean up;
# like scratch paper for audio chunks we cut from a long recording.
import tempfile
import threading

# `Callable[[int, int, str], None]` means "a function you can call with three
# arguments (two ints, one str) that returns nothing" — used for progress
# updates; like a doorbell the typist rings so the UI can show "50% done".
from collections.abc import Callable

# `@dataclass` auto-builds a simple class that mostly holds labeled fields;
# like a form with named blanks (plain text, timestamps, speakers, …).
from dataclasses import dataclass

# `Path` is a friendly way to handle file paths on any operating system;
# like an address written in a standard format so Windows and Mac agree.
from pathlib import Path

# Faster-Whisper loads the AI "listener" model efficiently; WHO USES IT: this
# file only — we create `WhisperModel` objects inside `_get_whisper_model`.
from faster_whisper import WhisperModel

# Numbers from app settings: max chunk length and when to start chunking;
# like recipe amounts shared from the app's config cookbook.
from app.config import (
    CHUNK_DURATION_SEC,
    EARLY_CHAT_CHUNKS,
    MIN_DURATION_FOR_CHUNKING,
    PARALLEL_TRANSCRIPTION_WORKERS,
    STREAM_CHUNK_DURATION_SEC,
)

# Logger named after this module so log lines show where they came from;
# WHO USES IT: this file's functions when they call logger.info(...).
logger = logging.getLogger(__name__)

# In-memory cache: one loaded AI model per unique (size, device, number style);
# like keeping one heavy dictionary on the desk instead of fetching a new copy
# from the library every time someone asks for a translation.
_whisper_cache: dict[str, WhisperModel] = {}


# Loads or reuses the AI "ears" (Whisper) for a given size/device; like borrowing
# the same pair of headphones instead of buying new ones each song.
# WHO CALLS THIS: `transcribe()` and `_transcribe_single()` only (this file).
def _get_whisper_model(model_size: str, device: str = "cpu", compute_type: str = "int8") -> WhisperModel:
    """Return a cached WhisperModel, loading it only on first use per config."""
    # Build a single text key so "small on CPU in int8" never collides with
    # "large on GPU in float16"; like a coat-check ticket combining three tags.
    key = f"{model_size}:{device}:{compute_type}"  # One string fingerprint for this hardware+model recipe.
    # First time we see this combo, load the model (slow); later times reuse it.
    if key not in _whisper_cache:
        # Inform operators in logs; does not change behavior, only visibility.
        logger.info("Loading Whisper model: %s (device=%s, compute=%s)", model_size, device, compute_type)
        # Store in the dict so the next call is instant; like filling the cache.
        _whisper_cache[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    # WHO CALLS THIS: only `transcribe()` and `_transcribe_single()` in this file.
    return _whisper_cache[key]


# Magic marker: Python auto-writes `__init__` and nice printing for simple
# "bag of fields" types; like a stamp that says "this class is just labeled data".
@dataclass
# The envelope we hand back to the rest of the app after listening to audio;
# WHO BUILDS INSTANCES: `_build_result()` (called from `_transcribe_single` and `transcribe`).
# WHO READS IT: `app/api/transcription.py` and any code that serializes the transcript to JSON.
class TranscriptResult:
    # All words run together in one string; easy for search or copy-paste.
    plain_text: str
    # Same content but each line shows start–end times; like subtitles with clocks.
    timestamped_text: str
    # List of small dicts: each piece has start, end, text (and maybe speaker).
    segments: list[dict]
    # Optional list of speaker labels seen ("Speaker A", …); None if not used.
    speakers: list[str] | None = None
    # Optional segments that have a speaker tag attached; None if diarization off.
    speaker_segments: list[dict] | None = None
    # If language was "auto", Whisper may fill this with what it guessed (e.g. "en").
    detected_language: str | None = None


# Turns 67.3 seconds into "01:07" or "1:00:07" text for humans; like writing
# timestamps on a cassette label.
# WHO CALLS THIS: `_build_result()` only (this file).
def _format_ts(seconds: float) -> str:
    # Drop fractions: humans read whole seconds in timestamps; like rounding to
    # the nearest second on a kitchen timer display.
    total = int(seconds)  # Whole seconds only; fractional part discarded for display.
    # Split total seconds into hours and leftover seconds; 3600 = seconds per hour.
    h, remainder = divmod(total, 3600)  # `divmod` returns (quotient, remainder) in one step.
    # Split remainder into minutes and seconds; 60 = seconds per minute.
    m, s = divmod(remainder, 60)  # Same idea: minutes and seconds from the leftover hour-less part.
    # If there are hours, show H:MM:SS; otherwise MM:SS — like a short vs long race clock.
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"  # `:02d` pads minutes/seconds with a leading zero.


def _get_audio_duration(audio_path: Path) -> float:
    # Build the command to ask ffprobe for metadata in JSON; `-v quiet` hushes noise.
    cmd = [
        "ffprobe",  # The media "inspector" program (must be installed on the server).
        "-v", "quiet",  # Less chatter on the terminal; we only want the JSON answer.
        "-print_format", "json",  # Ask for machine-readable output, not paragraphs.
        "-show_format", str(audio_path),  # Include the `format` section (holds duration).
    ]
    # Run ffprobe, capture printed text, require success (`check=True`); like
    # running a helper script and reading its printout into `result.stdout`.
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Parse JSON, dive into format.duration, convert to float seconds; like reading
    # "length: 125.3" from a label on the box.
    return float(json.loads(result.stdout)["format"]["duration"])  # Parse JSON text → dict → duration string → number.

# Asks "how long is this file?" using ffprobe; like glancing at the runtime printed on a DVD box.
# WHO CALLS THIS: `_split_audio()` and `transcribe()` (this file).

# Cuts a long recording into shorter files (or returns the original if short enough);
# like slicing a baguette for sandwiches.
# WHO CALLS THIS: `transcribe()` only when chunking is enabled (this file).
def _split_audio(audio_path: Path, chunk_duration: int = CHUNK_DURATION_SEC) -> list[tuple[Path, float]]:
    # Ask how long the whole file is; one number in seconds.
    duration = _get_audio_duration(audio_path)
    # If the file is not much longer than one chunk, skip splitting — like reading
    # a one-page letter in one go instead of cutting it into strips.
    if duration <= chunk_duration * 1.5:
        # Return the original path with offset 0.0 (whole file starts at time zero).
        return [(audio_path, 0.0)]

    # Create a new empty temp directory whose name starts with vcai-chunks-;
    # like a disposable tray for sliced audio pieces.
    tmp_dir = Path(tempfile.mkdtemp(prefix="vcai-chunks-"))
    # Will hold (path_to_chunk_file, start_time_in_original_recording) pairs.
    chunks: list[tuple[Path, float]] = []
    # Where the current slice begins in the original timeline (seconds).
    start = 0.0

    # Walk forward until we have covered the full duration; like a cookie cutter
    # moving along a long dough strip.
    while start < duration:
        # Do not pass the end of the file; last chunk may be shorter than chunk_duration.
        end = min(start + chunk_duration, duration)
        # Name chunks chunk_000.wav, chunk_001.wav, … so order is obvious.
        chunk_path = tmp_dir / f"chunk_{len(chunks):03d}.wav"
        # ffmpeg: copy stream without re-encoding between -ss and -to; fast but
        # needs keyframe-friendly cuts — acceptable for our chunking strategy.
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ss", str(start), "-to", str(end),
            "-c", "copy", str(chunk_path),
        ]
        # Run ffmpeg; fail loudly if something is wrong with the file or codecs.
        subprocess.run(cmd, capture_output=True, check=True)
        # Remember this file and where it sits on the original recording's clock.
        chunks.append((chunk_path, start))
        # Next slice starts where this one ended; no gap, no overlap.
        start = end

    # WHO CALLS THIS: only `transcribe()` when the recording is long enough to chunk.
    return chunks


# Takes raw timed text pieces and optional speaker labels → one `TranscriptResult` object;
# like assembling pages into a bound report.
# WHO CALLS THIS: `_transcribe_single()` and `transcribe()` only (this file).
def _build_result(raw_segments: list[dict], speaker_map: dict[int, str] | None = None) -> TranscriptResult:
    # If we have a map from segment index → speaker label, stamp each segment; like
    # writing names in the margin of a script.
    if speaker_map:
        # `enumerate` gives (0, first_segment), (1, second_segment), …
        for i, seg in enumerate(raw_segments):
            # Default to "Unknown" if an index is missing from the map.
            seg["speaker"] = speaker_map.get(i, "Unknown")

    # Two empty lists we will fill while walking segments; like two notepads.
    plain_parts, timestamped_parts = [], []
    # One pass: build both human-readable views from the same underlying pieces.
    for seg in raw_segments:
        # If this segment has a speaker key, prefix lines with [Speaker A] style.
        prefix = f"[{seg['speaker']}] " if seg.get("speaker") else ""
        # Plain line: optional speaker tag + spoken text.
        plain_parts.append(f"{prefix}{seg['text']}")
        # Timestamped line: clock range + same prefix + text; like subtitles export.
        timestamped_parts.append(
            f"[{_format_ts(seg['start'])} - {_format_ts(seg['end'])}] {prefix}{seg['text']}"
        )

    # Unique speaker names, sorted alphabetically, only where speaker exists; like
    # collecting name tags from a meeting without duplicates.
    speakers = sorted({s.get("speaker", "") for s in raw_segments if s.get("speaker")})

    # Package everything into the dataclass; empty speaker list becomes None for "unused".
    return TranscriptResult(
        plain_text=" ".join(plain_parts),  # Join all spoken lines with spaces → one paragraph-style string.
        timestamped_text="\n".join(timestamped_parts),  # One subtitle-style line per segment, stacked vertically.
        segments=raw_segments,  # Keep the structured list for APIs that need start/end/text dicts.
        speakers=speakers or None,  # `or None` turns an empty list into None (meaning "no speaker info").
        speaker_segments=[s for s in raw_segments if s.get("speaker")] or None,  # Only segments that carry a speaker key.
    )


# Guesses "who spoke when" using silence gaps only — simple, not courtroom-grade;
# like switching name tags whenever the room goes quiet for more than a beat.
# WHO CALLS THIS: `transcribe()` when `diarize=True` (this file only).
def diarize_simple(segments: list[dict], num_speakers: int = 2) -> dict[int, str]:
    """Heuristic speaker assignment based on >1.5 s pauses between segments."""
    # No segments → nothing to label; return empty map; like an empty attendance sheet.
    if not segments:
        return {}

    # Will map segment index (0, 1, 2, …) → "Speaker A", "Speaker B", …
    speaker_map: dict[int, str] = {}
    # Start everyone as speaker index 0 until a long pause suggests a switch.
    current_speaker = 0

    # Walk segments in order; pauses between Whisper pieces guide guesses.
    for i, seg in enumerate(segments):
        # If not the first segment and gap from previous end to this start > 1.5s,
        # assume someone new might be talking — rotate among num_speakers labels.
        if i > 0 and seg["start"] - segments[i - 1]["end"] > 1.5:
            # `% num_speakers` wraps: A→B→A for two speakers; like passing a talking stick.
            current_speaker = (current_speaker + 1) % num_speakers
        # chr(65) is 'A', 66 is 'B', … — classic "Speaker A" labeling.
        speaker_map[i] = f"Speaker {chr(65 + current_speaker)}"

    # WHO CALLS THIS: only `transcribe()` in this file (not imported elsewhere today).
    return speaker_map


# Interprets the user's language string (auto, fixed code, or "→ English" modes).
def _parse_language(language: str) -> tuple[str | None, str, bool]:
    """Return (whisper_language, whisper_task, needs_llm_translate).

    - "auto"       → (None, "transcribe", False)  – let Whisper detect
    - "xx"         → ("xx", "transcribe", False)   – force that language
    - "xx>en"      → ("xx", "transcribe", True)    – transcribe in source, then LLM translates
    - "auto>en"    → (None, "transcribe", True)    – detect + transcribe, then LLM translates
    """
    # WHO CALLS THIS: `app/api/transcription.py` (checks if LLM translation is needed), plus
    # `_transcribe_single()` and chunked `transcribe()` in this file (Whisper language/task).
    # Normalize user input: trim spaces, lowercase; empty becomes "auto"; like
    # cleaning a form field before reading it.
    lang = language.strip().lower() if language else "auto"
    # If user chose "something → English", we still transcribe in source language
    # here; a different part of the app (LLM) does English translation later.
    if ">en" in lang:
        # Remove the ">en" marker to recover the source language part.
        source = lang.replace(">en", "").strip()
        # "auto>en" or bare ">en" means detect language but still plan translation.
        if source in ("auto", ""):
            return None, "transcribe", True
        # e.g. "hi>en" → force Hindi for Whisper, flag that translation is wanted.
        return source, "transcribe", True
    # Plain auto or empty → let Whisper pick language, no translation flag.
    if lang in ("auto", ""):
        return None, "transcribe", False
    # Specific code like "en", "es" → lock Whisper to that language.
    return lang, "transcribe", False


# WHO READS THIS: `app/api/transcription.py` turns short codes into friendly names for users/logs.
LANGUAGE_NAMES: dict[str, str] = {
    # Short code → human-readable name for UI or messages; like a legend on a map.
    "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German",
    "ja": "Japanese", "zh": "Chinese", "ar": "Arabic", "pt": "Portuguese",
    "ru": "Russian", "ko": "Korean", "it": "Italian", "tr": "Turkish",
    "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "mr": "Marathi",
    "gu": "Gujarati", "ur": "Urdu",
}


# Runs Whisper on one whole audio file (no chunking); handles language retries and VAD fallback.
# WHO CALLS THIS: `transcribe()` only when the recording is short enough to skip chunking (this file).
def _transcribe_single(model: WhisperModel, audio_path: Path, language: str) -> TranscriptResult:
    # Unpack: lang for Whisper (None = auto), task always transcribe here, third
    # value ignored in this function (translation handled upstream in API layer).
    lang, task, _ = _parse_language(language)
    # Ask the model to listen; `beam_size` searches a bit wider for best text;
    # `vad_filter` tries to skip silent bits (Voice Activity Detection); like
    # skipping blank pages when OCR-ing a book.
    segments_iter, info = model.transcribe(
        str(audio_path), language=lang, task=task, beam_size=5, vad_filter=True,
    )

    # If language was auto and Whisper is not confident, retry forcing English;
    # like asking "was that really French?" and defaulting to English if unsure.
    if lang is None and info.language_probability < 0.6:
        # Log the guess and confidence percentage for debugging.
        logger.info(
            "Low-confidence language detection: '%s' (%.0f%%), retrying with English",
            info.language, info.language_probability * 100,
        )
        # Second pass: lock to English so garbled wrong-language output is less likely.
        segments_iter, info = model.transcribe(
            str(audio_path), language="en", task=task, beam_size=5, vad_filter=True,
        )

    # Convert model output objects into plain dicts Python can JSON-save later;
    # skip empty text; round times to 2 decimals for stable display.
    raw = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments_iter if s.text.strip()
    ]

    # Sometimes VAD is too aggressive and drops everything — retry without it.
    if not raw:
        logger.info("VAD filter produced no segments, retrying without VAD filter")
        # Prefer explicit lang; else model's detected lang; else English fallback.
        retry_lang = lang if lang is not None else (info.language if info.language else "en")
        segments_iter, info = model.transcribe(
            str(audio_path), language=retry_lang, task=task, beam_size=5, vad_filter=False,
        )
        # Same dict shaping as above after the retry pass.
        raw = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments_iter if s.text.strip()
        ]

    # Build user-facing strings and lists from the raw segment dicts.
    result = _build_result(raw)
    # If we started in auto mode and Whisper reported a language, store it on the result.
    if lang is None and info.language:
        result.detected_language = info.language
    # WHO CALLS THIS: only `transcribe()` in this file (short audio, no chunking path).
    return result


# Main door from the outside world: path in, `TranscriptResult` out — may chunk long files.
# WHO CALLS THIS: `app/api/transcription.py` (`_run_transcription_sync`) and `app/cli.py`.
def transcribe(
    audio_path: Path,
    model_size: str = "base",
    language: str = "en",
    progress_callback: Callable[[int, int, str], None] | None = None,
    diarize: bool = False,
    num_speakers: int = 2,
) -> TranscriptResult:
    from app.config import STT_PROVIDER

    if STT_PROVIDER == "aws":
        from app.services.aws_transcribe import transcribe_path_to_segments

        if progress_callback:
            progress_callback(0, 1, "Transcribing with Amazon Transcribe...")
        raw = transcribe_path_to_segments(audio_path, language)
        if progress_callback:
            progress_callback(1, 1, "Transcription complete")
        speaker_map = diarize_simple(raw, num_speakers) if diarize else None
        return _build_result(raw, speaker_map)

    # Need duration to decide chunking strategy; one ffprobe call.
    duration = _get_audio_duration(audio_path)
    # Compare to config threshold: long files get sliced; short ones stay whole.
    use_chunking = duration > MIN_DURATION_FOR_CHUNKING

    # Load or reuse Whisper for this model_size (default device/cpu settings inside).
    model = _get_whisper_model(model_size)

    # ----- Short file path: one shot, simpler logic -----
    if not use_chunking:
        # Optional progress: 0 of 1 steps, with a message; UI can show a spinner.
        if progress_callback:
            progress_callback(0, 1, "Transcribing audio...")
        # Run the full pipeline on the entire file at once.
        result = _transcribe_single(model, audio_path, language)
        # If caller wants speaker labels, re-wrap segments with diarization map.
        if diarize and result.segments:
            return _build_result(result.segments, diarize_simple(result.segments, num_speakers))
        # Otherwise return transcription without speaker guessing.
        return result

    # ----- Long file path: chunk, transcribe each, stitch timestamps -----
    # Parse language once for all chunks (may lock to detected lang after chunk 0).
    lang, task, _ = _parse_language(language)
    # Create chunk files (or single-item list if unexpectedly short after check).
    chunks = _split_audio(audio_path)
    # How many pieces to process — drives progress denominator.
    total = len(chunks)
    # All segment dicts from every chunk, with times shifted to global timeline.
    all_segments: list[dict] = []
    # Remember language we locked onto when auto-detecting from first chunk.
    detected_lang: str | None = None

    # Process each chunk file in order; index i is also used for progress messages.
    for i, (chunk_path, start_offset) in enumerate(chunks):
        # Tell the UI which slice we are on; like "page 3 of 10".
        if progress_callback:
            progress_callback(i, total, f"Transcribing chunk {i + 1}/{total}...")

        # Transcribe this slice only; same beam_size and VAD as single-file path.
        segments_iter, info = model.transcribe(
            str(chunk_path), language=lang, task=task, beam_size=5, vad_filter=True,
        )
        # On the first chunk only, if language was auto, adopt Whisper's guess if confident.
        if i == 0 and lang is None and info.language:
            # Low confidence → force English for remaining chunks so we stay consistent.
            if info.language_probability < 0.6:
                logger.info(
                    "Low-confidence language detection: '%s' (%.0f%%), falling back to English",
                    info.language, info.language_probability * 100,
                )
                lang = "en"
            else:
                # High confidence → stick with detected code for all following chunks.
                lang = info.language
            # Record what we decided for the final `TranscriptResult`.
            detected_lang = lang

        # Collect segments for this chunk before merging into the big list.
        chunk_segments = []
        # Iterate the generator from Whisper for this chunk's audio.
        for seg in segments_iter:
            # Ignore whitespace-only hallucinations.
            text = seg.text.strip()
            if text:
                # Shift times by start_offset so second chunk does not restart at 0:00;
                # like stitching tape measures end-to-end.
                chunk_segments.append({
                    "start": round(seg.start + start_offset, 2),
                    "end": round(seg.end + start_offset, 2),
                    "text": text,
                })

        # Same VAD-empty retry as single-file path, but per chunk.
        if not chunk_segments:
            logger.info("VAD filter produced no segments for chunk %d, retrying without VAD", i + 1)
            # Language should be set after chunk 0; if still None, English is safe default.
            retry_lang = lang if lang is not None else "en"
            segments_iter, info = model.transcribe(
                str(chunk_path), language=retry_lang, task=task, beam_size=5, vad_filter=False,
            )
            # Re-fill chunk_segments with the same offset math as above.
            for seg in segments_iter:
                text = seg.text.strip()
                if text:
                    chunk_segments.append({
                        "start": round(seg.start + start_offset, 2),
                        "end": round(seg.end + start_offset, 2),
                        "text": text,
                    })

        # Append this chunk's pieces to the full transcript timeline.
        all_segments.extend(chunk_segments)

    # Final progress tick: all chunks done; UI can show 100%.
    if progress_callback:
        progress_callback(total, total, "Transcription complete")

    # Either build speaker map from pauses or leave None for plain transcript.
    speaker_map = diarize_simple(all_segments, num_speakers) if diarize else None
    # Assemble plain text, timestamped text, and optional speaker fields.
    result = _build_result(all_segments, speaker_map)
    # Attach the language we pinned during chunked auto-detect (may be None if forced from start).
    result.detected_language = detected_lang
    # WHO CALLS THIS: `app/api/transcription.py` (`_run_transcription_sync`) and
    # `app/cli.py` — the main public API of this module.
    return result


# ── Streaming / parallel transcription ───────────────────────────────

_thread_local = threading.local()


def _get_thread_whisper_model(
    model_size: str, device: str = "cpu", compute_type: str = "int8"
) -> WhisperModel:
    """Return a thread-local WhisperModel so parallel workers never share state."""
    if not hasattr(_thread_local, "models"):
        _thread_local.models: dict[str, WhisperModel] = {}
    key = f"{model_size}:{device}:{compute_type}"
    if key not in _thread_local.models:
        logger.info(
            "Loading thread-local Whisper model: %s (device=%s)", model_size, device
        )
        _thread_local.models[key] = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
    return _thread_local.models[key]


def _transcribe_one_chunk(
    chunk_path: Path,
    start_offset: float,
    model_size: str,
    language: str,
) -> tuple[list[dict], str | None]:
    """Transcribe a single audio chunk using a thread-local model.

    Returns (segments_with_offset, detected_language_or_none).
    """
    model = _get_thread_whisper_model(model_size)
    lang, task, _ = _parse_language(language)

    segments_iter, info = model.transcribe(
        str(chunk_path), language=lang, task=task, beam_size=5, vad_filter=True
    )

    segments: list[dict] = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append(
                {
                    "start": round(seg.start + start_offset, 2),
                    "end": round(seg.end + start_offset, 2),
                    "text": text,
                }
            )

    if not segments:
        retry_lang = lang if lang is not None else "en"
        segments_iter, info = model.transcribe(
            str(chunk_path),
            language=retry_lang,
            task=task,
            beam_size=5,
            vad_filter=False,
        )
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                segments.append(
                    {
                        "start": round(seg.start + start_offset, 2),
                        "end": round(seg.end + start_offset, 2),
                        "text": text,
                    }
                )

    detected = info.language if lang is None and info.language else None
    return segments, detected


def transcribe_streaming(
    audio_path: Path,
    model_size: str = "base",
    language: str = "en",
    chunk_callback: Callable[[int, int, list[dict], bool], None] | None = None,
    diarize: bool = False,
    num_speakers: int = 2,
    chunk_duration: int = STREAM_CHUNK_DURATION_SEC,
    max_workers: int = PARALLEL_TRANSCRIPTION_WORKERS,
    early_chat_chunks: int = EARLY_CHAT_CHUNKS,
) -> TranscriptResult:
    """Streaming transcription with parallel workers and early partial results.

    Parameters
    ----------
    chunk_callback:
        Called after each chunk completes:
        ``(completed_count, total_chunks, ordered_segments_so_far, is_chat_ready)``
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from app.config import STT_PROVIDER

    if STT_PROVIDER == "aws":
        from app.services.aws_transcribe import transcribe_path_to_segments

        if chunk_callback:
            chunk_callback(0, 1, [], False)
        raw = transcribe_path_to_segments(audio_path, language)
        if chunk_callback:
            chunk_callback(1, 1, raw, True)
        if diarize and raw:
            return _build_result(raw, diarize_simple(raw, num_speakers))
        return _build_result(raw)

    duration = _get_audio_duration(audio_path)

    if duration <= chunk_duration * 2:
        model = _get_whisper_model(model_size)
        if chunk_callback:
            chunk_callback(0, 1, [], False)
        result = _transcribe_single(model, audio_path, language)
        if chunk_callback:
            chunk_callback(1, 1, result.segments, True)
        if diarize and result.segments:
            return _build_result(
                result.segments, diarize_simple(result.segments, num_speakers)
            )
        return result

    chunks = _split_audio(audio_path, chunk_duration)
    total = len(chunks)
    completed_chunks: dict[int, list[dict]] = {}
    detected_lang: str | None = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, (chunk_path, start_offset) in enumerate(chunks):
            future = executor.submit(
                _transcribe_one_chunk, chunk_path, start_offset, model_size, language
            )
            futures[future] = i

        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                segments, det_lang = future.result()
            except Exception:
                logger.warning("Chunk %d failed, skipping", chunk_idx, exc_info=True)
                completed_chunks[chunk_idx] = []
                continue

            completed_chunks[chunk_idx] = segments
            if det_lang and detected_lang is None:
                detected_lang = det_lang

            ordered_segments: list[dict] = []
            contiguous = 0
            for j in range(total):
                if j in completed_chunks:
                    ordered_segments.extend(completed_chunks[j])
                    contiguous += 1
                else:
                    break

            is_chat_ready = contiguous >= min(early_chat_chunks, total)

            if chunk_callback:
                chunk_callback(len(completed_chunks), total, ordered_segments, is_chat_ready)

    all_segments: list[dict] = []
    for i in range(total):
        all_segments.extend(completed_chunks.get(i, []))

    speaker_map = diarize_simple(all_segments, num_speakers) if diarize else None
    result = _build_result(all_segments, speaker_map)
    result.detected_language = detected_lang
    return result
