"""FFmpeg audio extract + Faster-Whisper with word timestamps, overlapping windows, and aligned segments."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.config import (
    CODEXVID_AUDIO_OVERLAP_SEC,
    CODEXVID_FINE_SEG_MAX_SEC,
    CODEXVID_FINE_SEG_MIN_SEC,
    CODEXVID_MAX_PARALLEL_WORKERS,
    CODEXVID_PARALLEL_WORKERS,
    CODEXVID_WHISPER_BEAM_SIZE,
    CODEXVID_WHISPER_CHUNK_SEC,
    STT_PROVIDER,
)
from app.codexvid.timestamp_utils import (
    dedupe_overlapping_words,
    merge_segments,
    normalize_transcript_segments,
    words_to_fine_segments,
)
from app.services.transcription import (
    _get_audio_duration,
    _get_whisper_model,
    _get_thread_whisper_model,
    _parse_language,
    _split_audio,
    _transcribe_one_chunk,
)

logger = logging.getLogger(__name__)


def _adaptive_windowing(duration: float, chunk_sec: float, overlap_sec: float) -> tuple[float, float]:
    """Increase window size for longer media to reduce ffmpeg/transcribe overhead."""
    if duration >= 12 * 60:
        return max(chunk_sec, 48.0), min(overlap_sec, 4.0)
    if duration >= 8 * 60:
        return max(chunk_sec, 40.0), min(overlap_sec, 4.0)
    return chunk_sec, overlap_sec


def extract_audio_wav(video_path: Path, out_wav: Path) -> None:
    """16 kHz mono WAV for Whisper."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _cleanup_chunk_paths(chunks: list[tuple[Path, float]], original_wav: Path) -> None:
    if not chunks:
        return
    first = chunks[0][0].resolve()
    orig = original_wav.resolve()
    if first == orig:
        return
    tmp_dir = first.parent
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except OSError:
        logger.warning("Could not remove chunk temp dir %s", tmp_dir)


def _split_audio_overlapping(
    audio_path: Path,
    *,
    chunk_duration: float,
    overlap_sec: float,
) -> tuple[list[tuple[Path, float]], Path | None]:
    """Slice WAV into overlapping windows. Returns (chunk_path, global_start_offset) and tmp_dir or None."""
    duration = _get_audio_duration(audio_path)
    step = max(chunk_duration - overlap_sec, chunk_duration * 0.4)
    if duration <= chunk_duration * 1.15:
        return [(audio_path, 0.0)], None

    tmp_dir = Path(tempfile.mkdtemp(prefix="codexvid-ov-"))
    chunks: list[tuple[Path, float]] = []
    start = 0.0
    idx = 0
    while start < duration - 0.05:
        end = min(start + chunk_duration, duration)
        chunk_path = tmp_dir / f"ov_{idx:03d}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            str(start),
            "-to",
            str(end),
            "-c",
            "copy",
            str(chunk_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        chunks.append((chunk_path, start))
        if end >= duration - 0.02:
            break
        start += step
        idx += 1
    return chunks, tmp_dir


def _segment_iter_to_word_list(
    segments_iter,
    start_offset: float,
) -> list[dict]:
    """Flatten faster-whisper segments into word dicts with absolute times."""
    words_flat: list[dict] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        w_attr = getattr(seg, "words", None)
        added = False
        if w_attr:
            for w in w_attr:
                wt = (getattr(w, "word", None) or "").strip()
                if not wt:
                    continue
                words_flat.append(
                    {
                        "word": wt,
                        "start": float(w.start) + start_offset,
                        "end": float(w.end) + start_offset,
                    }
                )
                added = True
        if not added:
            words_flat.append(
                {
                    "word": text,
                    "start": float(seg.start) + start_offset,
                    "end": float(seg.end) + start_offset,
                }
            )
    return words_flat


def _transcribe_window_words(
    chunk_path: Path,
    start_offset: float,
    model_size: str,
    language: str,
) -> tuple[list[dict], str | None]:
    """Transcribe one window with word timestamps; return word list + detected lang."""
    model = _get_thread_whisper_model(model_size)
    lang, task, _ = _parse_language(language)

    segments_iter, info = model.transcribe(
        str(chunk_path),
        language=lang,
        task=task,
        beam_size=CODEXVID_WHISPER_BEAM_SIZE,
        vad_filter=True,
        word_timestamps=True,
    )
    segments_list = list(segments_iter)
    words_flat = _segment_iter_to_word_list(iter(segments_list), start_offset)

    if not words_flat:
        segments_iter2, _info2 = model.transcribe(
            str(chunk_path),
            language=lang if lang is not None else "en",
            task=task,
            beam_size=CODEXVID_WHISPER_BEAM_SIZE,
            vad_filter=False,
            word_timestamps=True,
        )
        words_flat = _segment_iter_to_word_list(segments_iter2, start_offset)

    detected = info.language if lang is None and getattr(info, "language", None) else None
    return words_flat, detected


def _transcribe_whisper_parallel_overlapping(
    wav_path: Path,
    *,
    model_size: str,
    language: str,
    chunk_sec: float,
    overlap_sec: float,
    max_workers: int,
) -> list[dict]:
    """Overlapping windows + word dedupe + 2–5s fine segments + normalize."""
    duration = _get_audio_duration(wav_path)
    chunk_sec, overlap_sec = _adaptive_windowing(duration, chunk_sec, overlap_sec)
    workers_cap = max(1, min(max_workers, CODEXVID_MAX_PARALLEL_WORKERS, (os.cpu_count() or 1)))

    logger.info(
        "CodexVid Whisper: duration=%.1fs, chunk=%.1fs, overlap=%.1fs, workers=%d",
        duration,
        chunk_sec,
        overlap_sec,
        workers_cap,
    )

    chunks, ov_tmp = _split_audio_overlapping(
        wav_path,
        chunk_duration=chunk_sec,
        overlap_sec=overlap_sec,
    )

    try:
        if len(chunks) == 1:
            words, _ = _transcribe_window_words(
                chunks[0][0], chunks[0][1], model_size, language
            )
            words = dedupe_overlapping_words(words)
            fine = words_to_fine_segments(
                words,
                min_sec=CODEXVID_FINE_SEG_MIN_SEC,
                max_sec=CODEXVID_FINE_SEG_MAX_SEC,
            )
            if fine:
                return normalize_transcript_segments(fine)
            # Fallback: coarse segments
            model = _get_thread_whisper_model(model_size)
            lang, task, _ = _parse_language(language)
            segs_iter, _i = model.transcribe(
                str(wav_path),
                language=lang,
                task=task,
                beam_size=CODEXVID_WHISPER_BEAM_SIZE,
                vad_filter=True,
            )
            coarse = [
                {
                    "text": s.text.strip(),
                    "start": round(float(s.start), 2),
                    "end": round(float(s.end), 2),
                }
                for s in segs_iter
                if (s.text or "").strip()
            ]
            return normalize_transcript_segments(merge_segments(coarse))

        out_by_idx: dict[int, list[dict]] = {}
        n = len(chunks)
        workers = max(1, min(workers_cap, n))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    _transcribe_window_words,
                    chunk_path,
                    start_off,
                    model_size,
                    language,
                ): idx
                for idx, (chunk_path, start_off) in enumerate(chunks)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                words, _det = fut.result()
                out_by_idx[idx] = words

        all_words: list[dict] = []
        for idx in range(n):
            all_words.extend(out_by_idx.get(idx, []))

        all_words = dedupe_overlapping_words(all_words)
        fine = words_to_fine_segments(
            all_words,
            min_sec=CODEXVID_FINE_SEG_MIN_SEC,
            max_sec=CODEXVID_FINE_SEG_MAX_SEC,
        )
        if fine:
            return normalize_transcript_segments(fine)

        if ov_tmp is not None:
            shutil.rmtree(ov_tmp, ignore_errors=True)
            ov_tmp = None

        # Fallback: legacy parallel chunk segments (no words)
        legacy_chunks = _split_audio(wav_path, chunk_duration=int(chunk_sec))
        try:
            if len(legacy_chunks) == 1:
                segs, _ = _transcribe_one_chunk(
                    legacy_chunks[0][0], legacy_chunks[0][1], model_size, language
                )
                return normalize_transcript_segments(merge_segments(segs))
            out_legacy: dict[int, list[dict]] = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                fmap = {
                    executor.submit(
                        _transcribe_one_chunk, p, off, model_size, language
                    ): ix
                    for ix, (p, off) in enumerate(legacy_chunks)
                }
                for fut in as_completed(fmap):
                    ix = fmap[fut]
                    segs, _ = fut.result()
                    out_legacy[ix] = segs
            merged_legacy: list[dict] = []
            for ix in range(len(legacy_chunks)):
                merged_legacy.extend(out_legacy.get(ix, []))
            merged_legacy.sort(key=lambda s: s["start"])
            return normalize_transcript_segments(merge_segments(merged_legacy))
        finally:
            _cleanup_chunk_paths(legacy_chunks, wav_path)
    finally:
        if ov_tmp is not None:
            shutil.rmtree(ov_tmp, ignore_errors=True)


def transcribe_video(
    video_path: Path,
    *,
    model_size: str = "base",
    language: str = "en",
) -> list[dict]:
    """Transcribe a video file; return ``[{text, start, end, words?}, ...]`` (seconds)."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        extract_audio_wav(video_path, wav_path)

        if STT_PROVIDER == "aws":
            from app.services.aws_transcribe import transcribe_path_to_segments

            logger.info(
                "CodexVid transcribe: AWS Transcribe (model_size=%s ignored)",
                model_size,
            )
            raw = transcribe_path_to_segments(wav_path, language)
            logger.info(
                "CodexVid transcribe: %d segments from %s", len(raw), video_path.name
            )
            return normalize_transcript_segments(merge_segments(raw))

        raw = _transcribe_whisper_parallel_overlapping(
            wav_path,
            model_size=model_size,
            language=language,
            chunk_sec=float(CODEXVID_WHISPER_CHUNK_SEC),
            overlap_sec=float(CODEXVID_AUDIO_OVERLAP_SEC),
            max_workers=CODEXVID_PARALLEL_WORKERS,
        )

        if not raw:
            logger.info("CodexVid: no segments; retrying full-file Whisper")
            model = _get_whisper_model(model_size)
            lang, task, _ = _parse_language(language)
            segments_iter, _info = model.transcribe(
                str(wav_path),
                language=lang,
                task=task,
                beam_size=CODEXVID_WHISPER_BEAM_SIZE,
                vad_filter=False,
                word_timestamps=True,
            )
            words = _segment_iter_to_word_list(segments_iter, 0.0)
            words = dedupe_overlapping_words(words)
            fine = words_to_fine_segments(
                words,
                min_sec=CODEXVID_FINE_SEG_MIN_SEC,
                max_sec=CODEXVID_FINE_SEG_MAX_SEC,
            )
            if fine:
                raw = normalize_transcript_segments(fine)
            else:
                segments_iter2, _ = model.transcribe(
                    str(wav_path),
                    language=lang,
                    task=task,
                    beam_size=CODEXVID_WHISPER_BEAM_SIZE,
                    vad_filter=False,
                )
                raw = [
                    {
                        "start": round(s.start, 2),
                        "end": round(s.end, 2),
                        "text": s.text.strip(),
                    }
                    for s in segments_iter2
                    if (s.text or "").strip()
                ]
                raw = normalize_transcript_segments(merge_segments(raw))

        logger.info(
            "CodexVid transcribe: %d segments from %s", len(raw), video_path.name
        )
        return raw
    finally:
        wav_path.unlink(missing_ok=True)
