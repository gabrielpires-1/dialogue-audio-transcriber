from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimedWord:
    start: float
    end: float
    text: str

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker_id: str


def _distance_to_turn(point: float, turn: SpeakerTurn) -> float:
    if point < turn.start:
        return turn.start - point
    if point >= turn.end:
        return point - turn.end
    return 0.0


def speaker_at(midpoint: float, turns: list[SpeakerTurn]) -> str:
    if not turns:
        return "SPEAKER_00"
    for turn in turns:
        if turn.start <= midpoint < turn.end:
            return turn.speaker_id
    return min(turns, key=lambda turn: _distance_to_turn(midpoint, turn)).speaker_id


def speaker_for_span(start: float, end: float, turns: list[SpeakerTurn]) -> str:
    if not turns:
        return "SPEAKER_00"
    best_id = turns[0].speaker_id
    best_overlap = -1.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = turn.speaker_id
    if best_overlap > 0:
        return best_id
    return speaker_at((start + end) / 2, turns)


def assign_speakers(
    words: list[TimedWord], turns: list[SpeakerTurn]
) -> list[tuple[TimedWord, str]]:
    return [(word, speaker_for_span(word.start, word.end, turns)) for word in words]


def _join_word_texts(texts: list[str]) -> str:
    return "".join(texts).strip()


def group_consecutive(
    assigned: list[tuple[TimedWord, str]],
) -> list[tuple[float, str, str]]:
    if not assigned:
        return []

    groups: list[tuple[float, str, str]] = []
    current_speaker = assigned[0][1]
    current_start = assigned[0][0].start
    texts = [assigned[0][0].text]

    for word, speaker in assigned[1:]:
        if speaker == current_speaker:
            texts.append(word.text)
            continue
        groups.append((current_start, current_speaker, _join_word_texts(texts)))
        current_speaker = speaker
        current_start = word.start
        texts = [word.text]

    groups.append((current_start, current_speaker, _join_word_texts(texts)))
    return groups


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _person_label(speaker_id: str, label_map: dict[str, str]) -> str:
    if speaker_id not in label_map:
        next_index = min(len(label_map) + 1, 2)
        label_map[speaker_id] = f"PERSON {next_index}"
    return label_map[speaker_id]


def format_transcript(words: list[TimedWord], turns: list[SpeakerTurn]) -> str:
    assigned = assign_speakers(words, turns)
    grouped = group_consecutive(assigned)
    label_map: dict[str, str] = {}
    lines: list[str] = []
    for start, speaker_id, text in grouped:
        if not text:
            continue
        label = _person_label(speaker_id, label_map)
        lines.append(f"[{format_timestamp(start)}] [{label}] - {text}")
    return "\n".join(lines)
