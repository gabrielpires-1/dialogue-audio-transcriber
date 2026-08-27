from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import UploadFile

from dialogue_audio_transcriber.config import Settings
from dialogue_audio_transcriber.constants import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    DIARIZATION_API,
    DIARIZATION_LOCAL,
    MAX_UPLOAD_SIZE_BYTES,
    READ_CHUNK_SIZE,
    TIMESTAMP_FORMAT,
    TRANSCRIPT_NOT_FOUND_MESSAGE,
    TRANSCRIPTION_FAILURE_MESSAGE,
)
from dialogue_audio_transcriber.diarization import resolve_diarization_mode
from dialogue_audio_transcriber.exceptions import (
    InvalidAudioFileError,
    TranscriptionError,
    TranscriptNotFoundError,
)
from dialogue_audio_transcriber.speaker_turns import TimedWord, format_transcript

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

    from dialogue_audio_transcriber.diarization import Diarizer

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(
        self, model: WhisperModel, diarizer: Diarizer, settings: Settings
    ) -> None:
        self._model = model
        self._diarizer = diarizer
        self._api_diarizer: Diarizer | None = None
        self._api_lock = threading.Lock()
        self._pyannote_api_key = settings.pyannote_api_key
        self._language = settings.language
        self._beam_size = settings.beam_size
        self._output_dir = settings.output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_diarization_mode(self, diarization: str) -> str:
        return resolve_diarization_mode(diarization, self._pyannote_api_key)

    def validate_upload(self, upload: UploadFile) -> None:
        filename = (upload.filename or "").strip()
        if not filename:
            raise InvalidAudioFileError("Missing audio file")

        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            raise InvalidAudioFileError(
                f"Unsupported file extension: {extension or 'none'}"
            )

        content_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise InvalidAudioFileError(
                f"Unsupported content type: {upload.content_type or 'none'}"
            )

    async def create_job(self, upload: UploadFile) -> tuple[str, Path]:
        self.validate_upload(upload)

        filename = (upload.filename or "").strip()
        extension = Path(filename).suffix.lower().lstrip(".")
        content = await self._read_upload(upload)

        timestamp = datetime.now(UTC).strftime(TIMESTAMP_FORMAT)
        audio_path = self._output_dir / f"{timestamp}.{extension}"
        result_filename = f"{timestamp}.txt"

        try:
            audio_path.write_bytes(content)
        except OSError as exc:
            raise TranscriptionError("Failed to persist uploaded audio") from exc

        return result_filename, audio_path

    def read_transcript(self, filename: str) -> str:
        path = self._resolve_transcript_path(filename)
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TranscriptionError("Failed to read transcript") from exc

    async def transcribe_and_write(
        self,
        audio_path: Path,
        result_filename: str,
        diarization: str = DIARIZATION_LOCAL,
    ) -> None:
        try:
            try:
                text = await asyncio.to_thread(
                    self._transcribe, audio_path, diarization
                )
            except Exception:
                logger.exception("Transcription failed for %s", result_filename)
                text = TRANSCRIPTION_FAILURE_MESSAGE
            self._write_transcript(result_filename, text)
        except Exception:
            logger.exception("Failed to write transcript %s", result_filename)
            try:
                self._write_transcript(result_filename, TRANSCRIPTION_FAILURE_MESSAGE)
            except Exception:
                logger.exception(
                    "Failed to write transcription error message for %s",
                    result_filename,
                )
        finally:
            audio_path.unlink(missing_ok=True)

    async def _read_upload(self, upload: UploadFile) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await upload.read(READ_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE_BYTES:
                raise InvalidAudioFileError(
                    f"File exceeds maximum size of {MAX_UPLOAD_SIZE_BYTES} bytes"
                )
            chunks.append(chunk)

        if total == 0:
            raise InvalidAudioFileError("Empty audio file")

        return b"".join(chunks)

    def _transcribe(self, audio_path: Path, diarization: str) -> str:
        from dialogue_audio_transcriber.diarization import release_gpu_cache

        spans = self._transcribe_segments(audio_path)
        release_gpu_cache()
        turns = self._diarizer_for(diarization).diarize(audio_path)
        return format_transcript(spans, turns)

    def _diarizer_for(self, diarization: str) -> Diarizer:
        if diarization != DIARIZATION_API:
            return self._diarizer
        if self._api_diarizer is None:
            with self._api_lock:
                if self._api_diarizer is None:
                    from dialogue_audio_transcriber.diarization import load_api_diarizer

                    self._api_diarizer = load_api_diarizer(self._pyannote_api_key)
        api_diarizer = self._api_diarizer
        if api_diarizer is None:
            raise TranscriptionError("Failed to load API diarizer")
        return api_diarizer

    def _transcribe_segments(self, audio_path: Path) -> list[TimedWord]:
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=self._language,
            beam_size=self._beam_size,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        spans: list[TimedWord] = []
        for segment in segments:
            text = segment.text or ""
            if not text.strip():
                continue
            spans.append(
                TimedWord(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                )
            )
        return spans

    def _write_transcript(self, result_filename: str, text: str) -> None:
        path = self._output_dir / result_filename
        payload = text if text.endswith("\n") else f"{text}\n"
        path.write_text(payload, encoding="utf-8")

    def _resolve_transcript_path(self, filename: str) -> Path:
        name = Path(filename).name
        if not name or name != filename or Path(name).suffix.lower() != ".txt":
            raise TranscriptNotFoundError(TRANSCRIPT_NOT_FOUND_MESSAGE)

        path = (self._output_dir / name).resolve()
        output_dir = self._output_dir.resolve()
        if not path.is_relative_to(output_dir) or not path.is_file():
            raise TranscriptNotFoundError(TRANSCRIPT_NOT_FOUND_MESSAGE)

        return path
