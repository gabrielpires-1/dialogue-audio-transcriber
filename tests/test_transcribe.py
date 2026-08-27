from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from dialogue_audio_transcriber.config import Settings
from dialogue_audio_transcriber.constants import (
    INVALID_DIARIZATION_MESSAGE,
    MISSING_PYANNOTE_API_KEY_MESSAGE,
)
from dialogue_audio_transcriber.diarization import PyannoteApiDiarizer
from dialogue_audio_transcriber.main import app, get_service
from dialogue_audio_transcriber.service import TranscriptionService
from dialogue_audio_transcriber.speaker_turns import (
    SpeakerTurn,
    TimedWord,
    format_transcript,
)


class FakeService:
    def __init__(self, api_key: str = "") -> None:
        from dialogue_audio_transcriber.diarization import resolve_diarization_mode

        self._api_key = api_key
        self._resolve = resolve_diarization_mode
        self.create_job_calls = 0
        self.transcribe_modes: list[str] = []

    def resolve_diarization_mode(self, diarization: str) -> str:
        return self._resolve(diarization, self._api_key)

    async def create_job(self, _upload: object) -> tuple[str, Path]:
        self.create_job_calls += 1
        return "20260826_000000_000000.txt", Path("recordings/audio.wav")

    async def transcribe_and_write(
        self,
        _audio_path: Path,
        _result_filename: str,
        diarization: str = "local",
    ) -> None:
        self.transcribe_modes.append(diarization)


class TranscribeEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeService()
        app.dependency_overrides[get_service] = lambda: self.service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def _post(self, query: str = "") -> object:
        path = "/transcribe" if not query else f"/transcribe?{query}"
        return self.client.post(
            path,
            files={"file": ("clip.wav", b"fake-audio", "audio/wav")},
        )

    def test_omitted_query_is_accepted_as_local(self) -> None:
        response = self._post()
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.service.create_job_calls, 1)
        self.assertEqual(self.service.transcribe_modes, ["local"])

    def test_explicit_local_is_accepted(self) -> None:
        response = self._post("diarization=local")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.service.transcribe_modes, ["local"])

    def test_invalid_diarization_returns_400_without_job(self) -> None:
        response = self._post("diarization=foo")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": INVALID_DIARIZATION_MESSAGE})
        self.assertEqual(self.service.create_job_calls, 0)
        self.assertEqual(self.service.transcribe_modes, [])

    def test_api_without_key_returns_400_without_job(self) -> None:
        response = self._post("diarization=api")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": MISSING_PYANNOTE_API_KEY_MESSAGE})
        self.assertEqual(self.service.create_job_calls, 0)
        self.assertEqual(self.service.transcribe_modes, [])

    def test_api_with_key_is_accepted(self) -> None:
        self.service = FakeService(api_key="secret")
        app.dependency_overrides[get_service] = lambda: self.service
        response = self._post("diarization=api")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.service.create_job_calls, 1)
        self.assertEqual(self.service.transcribe_modes, ["api"])


class TranscriptFormatTests(unittest.TestCase):
    def test_local_and_api_diarizers_write_the_same_person_lines(self) -> None:
        words_text = " hello"
        turns = [SpeakerTurn(0.0, 1.0, "SPEAKER_00")]
        local_diarizer = MagicMock()
        local_diarizer.diarize.return_value = turns
        api_diarizer = MagicMock()
        api_diarizer.diarize.return_value = turns
        model = MagicMock()
        model.transcribe.return_value = (
            [SimpleNamespace(start=0.0, end=1.0, text=words_text)],
            None,
        )

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                output_dir=Path(tmp),
                hf_token="hf",
                pyannote_api_key="secret",
            )
            service = TranscriptionService(
                model=model, diarizer=local_diarizer, settings=settings
            )
            service._api_diarizer = api_diarizer
            audio_path = Path(tmp) / "clip.wav"
            audio_path.write_bytes(b"fake")

            local_text = service._transcribe(audio_path, "local")
            api_text = service._transcribe(audio_path, "api")

        expected = "[00:00:00] [PERSON 1] - hello"
        self.assertEqual(local_text, expected)
        self.assertEqual(api_text, expected)
        self.assertEqual(local_text, api_text)
        local_diarizer.diarize.assert_called_once_with(audio_path)
        api_diarizer.diarize.assert_called_once_with(audio_path)

    def test_api_diarizer_turns_format_like_local_transcript(self) -> None:
        class FakeTurn:
            def __init__(self, start: float, end: float) -> None:
                self.start = start
                self.end = end

        class FakeOutput:
            def __init__(self) -> None:
                self.speaker_diarization = [
                    (FakeTurn(0.0, 1.0), "SPEAKER_00"),
                    (FakeTurn(1.0, 2.0), "SPEAKER_01"),
                ]

        class FakePipeline:
            def apply(self, _file: dict[str, str], num_speakers: int | None = None):
                return FakeOutput()

        diarizer = PyannoteApiDiarizer(pipeline=FakePipeline())
        turns = diarizer.diarize(Path("clip.wav"))
        text = format_transcript(
            [
                TimedWord(0.0, 1.0, " hello"),
                TimedWord(1.0, 2.0, " there"),
            ],
            turns,
        )
        self.assertEqual(
            text,
            "[00:00:00] [PERSON 1] - hello\n[00:00:01] [PERSON 2] - there",
        )


if __name__ == "__main__":
    unittest.main()
