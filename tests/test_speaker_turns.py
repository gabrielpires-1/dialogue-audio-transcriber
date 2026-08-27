from __future__ import annotations

import unittest

from dialogue_audio_transcriber.speaker_turns import (
    SpeakerTurn,
    TimedWord,
    format_timestamp,
    format_transcript,
    speaker_at,
    speaker_for_span,
)


class FormatTimestampTests(unittest.TestCase):
    def test_zero_pads_hours_minutes_seconds(self) -> None:
        self.assertEqual(format_timestamp(0), "00:00:00")
        self.assertEqual(format_timestamp(8.9), "00:00:08")
        self.assertEqual(format_timestamp(75), "00:01:15")
        self.assertEqual(format_timestamp(3661), "01:01:01")


class SpeakerAtTests(unittest.TestCase):
    def test_word_in_gap_uses_nearest_turn(self) -> None:
        turns = [
            SpeakerTurn(0.0, 1.0, "A"),
            SpeakerTurn(3.0, 4.0, "B"),
        ]
        self.assertEqual(speaker_at(1.75, turns), "A")
        self.assertEqual(speaker_at(2.6, turns), "B")

    def test_no_turns_defaults_to_first_speaker(self) -> None:
        self.assertEqual(speaker_at(1.0, []), "SPEAKER_00")


class SpeakerForSpanTests(unittest.TestCase):
    def test_assigns_speaker_with_greatest_overlap(self) -> None:
        turns = [
            SpeakerTurn(0.0, 0.4, "A"),
            SpeakerTurn(0.4, 2.0, "B"),
        ]
        self.assertEqual(speaker_for_span(0.0, 2.0, turns), "B")

    def test_gap_span_uses_nearest_turn(self) -> None:
        turns = [
            SpeakerTurn(0.0, 1.0, "A"),
            SpeakerTurn(3.0, 4.0, "B"),
        ]
        self.assertEqual(speaker_for_span(1.5, 2.0, turns), "A")


class FormatTranscriptTests(unittest.TestCase):
    def test_two_speakers_with_mid_utterance_switch(self) -> None:
        words = [
            TimedWord(0.0, 0.4, " hello"),
            TimedWord(0.4, 0.8, " there"),
            TimedWord(1.0, 1.4, " hi"),
            TimedWord(1.4, 1.8, " back"),
        ]
        turns = [
            SpeakerTurn(0.0, 0.9, "A"),
            SpeakerTurn(0.9, 2.0, "B"),
        ]
        self.assertEqual(
            format_transcript(words, turns),
            "[00:00:00] [PERSON 1] - hello there\n[00:00:01] [PERSON 2] - hi back",
        )

    def test_cloud_speaker_ids_use_same_person_labels(self) -> None:
        words = [
            TimedWord(0.0, 1.0, " hello"),
            TimedWord(1.0, 2.0, " there"),
        ]
        turns = [
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn(1.0, 2.0, "SPEAKER_01"),
        ]
        self.assertEqual(
            format_transcript(words, turns),
            "[00:00:00] [PERSON 1] - hello\n[00:00:01] [PERSON 2] - there",
        )

    def test_gap_word_stays_with_nearest_speaker(self) -> None:
        words = [
            TimedWord(0.0, 0.5, " hello"),
            TimedWord(1.5, 2.0, " gap"),
            TimedWord(3.0, 3.5, " after"),
        ]
        turns = [
            SpeakerTurn(0.0, 1.0, "A"),
            SpeakerTurn(3.0, 4.0, "B"),
        ]
        self.assertEqual(
            format_transcript(words, turns),
            "[00:00:00] [PERSON 1] - hello gap\n[00:00:03] [PERSON 2] - after",
        )

    def test_first_talker_is_person_one(self) -> None:
        words = [
            TimedWord(0.0, 1.0, " one"),
            TimedWord(1.0, 2.0, " two"),
        ]
        turns = [
            SpeakerTurn(0.0, 1.0, "B"),
            SpeakerTurn(1.0, 2.0, "A"),
        ]
        self.assertEqual(
            format_transcript(words, turns),
            "[00:00:00] [PERSON 1] - one\n[00:00:01] [PERSON 2] - two",
        )

    def test_empty_words_yields_empty_transcript(self) -> None:
        self.assertEqual(format_transcript([], []), "")

    def test_consecutive_segments_keep_whisper_spacing(self) -> None:
        words = [
            TimedWord(0.0, 1.0, " O que é que tu acha"),
            TimedWord(1.0, 2.0, " vai funcionar?"),
        ]
        turns = [SpeakerTurn(0.0, 2.0, "A")]
        self.assertEqual(
            format_transcript(words, turns),
            "[00:00:00] [PERSON 1] - O que é que tu acha vai funcionar?",
        )


if __name__ == "__main__":
    unittest.main()
