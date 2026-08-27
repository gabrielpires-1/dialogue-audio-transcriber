from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dialogue_audio_transcriber.config import _read_keys_from_env_file
from dialogue_audio_transcriber.constants import (
    INVALID_DIARIZATION_MESSAGE,
    MISSING_PYANNOTE_API_KEY_MESSAGE,
)
from dialogue_audio_transcriber.diarization import resolve_diarization_mode
from dialogue_audio_transcriber.exceptions import (
    InvalidDiarizationModeError,
    MissingPyannoteApiKeyError,
)


class ResolveDiarizationModeTests(unittest.TestCase):
    def test_local_is_accepted(self) -> None:
        self.assertEqual(resolve_diarization_mode("local", ""), "local")

    def test_api_is_accepted_when_key_is_present(self) -> None:
        self.assertEqual(resolve_diarization_mode("api", "secret"), "api")

    def test_invalid_value_fails_fast(self) -> None:
        with self.assertRaises(InvalidDiarizationModeError) as caught:
            resolve_diarization_mode("foo", "secret")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.detail, INVALID_DIARIZATION_MESSAGE)

    def test_empty_and_uppercase_values_fail_fast(self) -> None:
        for value in ("", "API", "Local", " cloud "):
            with (
                self.subTest(value=value),
                self.assertRaises(InvalidDiarizationModeError),
            ):
                resolve_diarization_mode(value, "secret")

    def test_api_without_key_fails_fast(self) -> None:
        with self.assertRaises(MissingPyannoteApiKeyError) as caught:
            resolve_diarization_mode("api", "")
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.detail, MISSING_PYANNOTE_API_KEY_MESSAGE)

    def test_api_with_whitespace_key_fails_fast(self) -> None:
        with self.assertRaises(MissingPyannoteApiKeyError):
            resolve_diarization_mode("api", "   ")


class DotenvApiKeyTests(unittest.TestCase):
    def test_reads_pyannoteai_api_key_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("PYANNOTEAI_API_KEY=from-file\n", encoding="utf-8")
            self.assertEqual(
                _read_keys_from_env_file(path, ("PYANNOTEAI_API_KEY",)),
                "from-file",
            )

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(
            _read_keys_from_env_file(Path("/tmp/does-not-exist.env"), ("HF_TOKEN",)),
            "",
        )


if __name__ == "__main__":
    unittest.main()
