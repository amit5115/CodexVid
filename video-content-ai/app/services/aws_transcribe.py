"""Amazon Transcribe (batch): S3 upload, job, poll, parse to Whisper-compatible segments."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import boto3
from botocore.exceptions import ClientError

from app.config import (
    AWS_REGION,
    AWS_TRANSCRIBE_BUCKET,
    AWS_TRANSCRIBE_POLL_TIMEOUT_SEC,
    STT_PROVIDER,
)
from app.services.transcription import _parse_language

logger = logging.getLogger(__name__)

# Short code → AWS LanguageCode (batch). Unknown codes fall back to en-US with a log line.
_AWS_LANG: dict[str, str] = {
    "en": "en-US",
    "es": "es-US",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-BR",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "zh": "zh-CN",
    "hi": "hi-IN",
    "ar": "ar-SA",
    "ru": "ru-RU",
    "tr": "tr-TR",
    "nl": "nl-NL",
    "pl": "pl-PL",
}

# When IdentifyLanguage is used, AWS allows up to 5 language options.
_IDENTIFY_OPTIONS = ["en-US", "es-US", "hi-IN", "fr-FR", "de-DE"]

_MEDIA_FORMAT = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".mp4": "mp4",
    ".m4a": "mp4",
    ".flac": "flac",
    ".ogg": "ogg",
    ".webm": "webm",
    ".mov": "mp4",
}


def _media_format_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    return _MEDIA_FORMAT.get(ext, "wav")


def _language_to_aws_settings(language: str) -> tuple[dict[str, Any], str | None]:
    """Build TranscriptionJob `Settings` / top-level LanguageCode for StartTranscriptionJob."""
    lang, _task, _trans = _parse_language(language)
    if lang is None:
        return (
            {
                "IdentifyLanguage": True,
                "LanguageOptions": _IDENTIFY_OPTIONS,
            },
            None,
        )
    code = _AWS_LANG.get(lang)
    if code is None:
        logger.warning("Unknown language code %r for AWS Transcribe; using en-US", lang)
        code = "en-US"
    return ({}, code)


def parse_transcript_json_to_segments(data: dict[str, Any], *, max_span_sec: float = 28.0) -> list[dict]:
    """Convert AWS Transcribe JSON (results.items) to [{start, end, text}, ...]."""
    items = data.get("results", {}).get("items", [])
    words: list[tuple[float, float, str]] = []
    for item in items:
        itype = item.get("type")
        alts = item.get("alternatives") or [{}]
        content = (alts[0].get("content") or "").strip()
        if itype == "pronunciation":
            if not content:
                continue
            st = float(item["start_time"])
            et = float(item["end_time"])
            words.append((st, et, content))
        elif itype == "punctuation" and content and words:
            st, et, prev = words[-1]
            words[-1] = (st, et, prev + content)

    if not words:
        return []

    segments: list[dict] = []
    i = 0
    while i < len(words):
        st0, et0, t0 = words[i]
        chunk_start = st0
        chunk_end = et0
        texts: list[str] = [t0]
        i += 1
        while i < len(words) and (words[i][1] - chunk_start) <= max_span_sec:
            _st, et, tx = words[i]
            chunk_end = et
            texts.append(tx)
            i += 1
        segments.append(
            {
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2),
                "text": " ".join(texts),
            }
        )
    return segments


def transcribe_path_to_segments(audio_path: Path, language: str) -> list[dict]:
    """Upload ``audio_path`` to S3, run Transcribe, return segments. Deletes S3 input object after."""
    if not AWS_TRANSCRIBE_BUCKET:
        raise RuntimeError(
            "VCAI_AWS_TRANSCRIBE_BUCKET is not set; required when VCAI_STT_PROVIDER=aws",
        )
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    job_name = f"vcai-{uuid.uuid4().hex[:24]}"
    key_in = f"input/{job_name}{audio_path.suffix.lower() or '.wav'}"
    s3_uri = f"s3://{AWS_TRANSCRIBE_BUCKET}/{key_in}"
    media_fmt = _media_format_for_path(audio_path)

    transcribe = boto3.client("transcribe", region_name=AWS_REGION)
    s3 = boto3.client("s3", region_name=AWS_REGION)

    logger.info("AWS Transcribe: uploading %s to %s", audio_path.name, s3_uri)
    s3.upload_file(str(audio_path), AWS_TRANSCRIBE_BUCKET, key_in)

    settings, language_code = _language_to_aws_settings(language)
    params: dict[str, Any] = {
        "TranscriptionJobName": job_name,
        "Media": {"MediaFileUri": s3_uri},
        "MediaFormat": media_fmt,
        "OutputBucketName": AWS_TRANSCRIBE_BUCKET,
    }
    if language_code:
        params["LanguageCode"] = language_code
    if settings:
        params["Settings"] = settings

    try:
        transcribe.start_transcription_job(**params)
    except ClientError as e:
        try:
            s3.delete_object(Bucket=AWS_TRANSCRIBE_BUCKET, Key=key_in)
        except OSError:
            pass
        raise RuntimeError(f"AWS StartTranscriptionJob failed: {e}") from e

    deadline = time.monotonic() + AWS_TRANSCRIBE_POLL_TIMEOUT_SEC
    uri: str | None = None
    while time.monotonic() < deadline:
        resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job = resp["TranscriptionJob"]
        status = job["TranscriptionJobStatus"]
        if status == "COMPLETED":
            uri = job.get("Transcript", {}).get("TranscriptFileUri")
            break
        if status == "FAILED":
            reason = job.get("FailureReason", "unknown")
            raise RuntimeError(f"AWS Transcribe job failed: {reason}")
        time.sleep(3.0)

    try:
        s3.delete_object(Bucket=AWS_TRANSCRIBE_BUCKET, Key=key_in)
    except ClientError:
        logger.warning("Could not delete S3 input object %s", key_in)

    if not uri:
        raise TimeoutError(
            f"AWS Transcribe job {job_name} did not complete within {AWS_TRANSCRIBE_POLL_TIMEOUT_SEC}s",
        )

    raw = urlopen(uri).read().decode("utf-8")  # noqa: S310 — AWS-signed HTTPS URL from API
    data = json.loads(raw)
    segments = parse_transcript_json_to_segments(data)
    logger.info("AWS Transcribe: %d segments from job %s", len(segments), job_name)
    return segments


def stt_provider_is_aws() -> bool:
    return STT_PROVIDER == "aws"
