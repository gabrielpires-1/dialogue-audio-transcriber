"""Generate a two-speaker Portuguese dialogue for transcription tests.

Uses Microsoft Edge neural voices. Turns can overlap so both speakers
talk at the same time, which is useful for diarization tests.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import edge_tts
import miniaudio

OUTPUT_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 16_000
GAP_SECONDS = 0.28
MAX_SAMPLE = 32767
MIN_SAMPLE = -32768

VOICES = {
    "pt-PT": ("pt-PT-DuarteNeural", "pt-PT-RaquelNeural"),
    "pt-BR": ("pt-BR-AntonioNeural", "pt-BR-FranciscaNeural"),
}


@dataclass(frozen=True)
class Turn:
    speaker: int
    text: str
    overlap: float = 0.0
    gap: float = GAP_SECONDS


DIALOGUES = {
    "pt-PT": [
        Turn(0, "Olá, já acabaste aquele relatório para a reunião de amanhã?"),
        Turn(1, "Quase! Só preciso de acrescentar os gráficos finais e fica pronto."),
        Turn(0, "Ótimo, porque o João precisa de rever antes da reunião."),
        Turn(1, "Sim, sim, mando ainda hoje à tarde!", overlap=1.15),
        Turn(0, "O que é que tu achas, vai funcionar?"),
        Turn(1, "Acho que sim!", overlap=0.75),
        Turn(0, "Os números do trimestre passado estão consistentes, não estão?"),
        Turn(1, "Pois, estão sim.", overlap=1.45),
        Turn(0, "Então fechamos com esses gráficos e eu apresento amanhã."),
        Turn(1, "Perfeito, pode deixar comigo.", overlap=0.9),
    ],
    "pt-BR": [
        Turn(0, "Oi, você já terminou aquele relatório para a reunião de amanhã?"),
        Turn(1, "Quase! Só preciso adicionar os gráficos finais e vai estar pronto."),
        Turn(0, "Ótimo, porque o João precisa revisar antes da reunião."),
        Turn(1, "Sim, sim, eu mando ainda hoje à tarde!", overlap=1.15),
        Turn(0, "O que você acha, vai funcionar?"),
        Turn(1, "Acho que sim!", overlap=0.75),
        Turn(0, "Os números do trimestre passado estão consistentes, né?"),
        Turn(1, "Uhum, estão sim.", overlap=1.45),
        Turn(0, "Então fechamos com esses gráficos e eu apresento amanhã."),
        Turn(1, "Perfeito, pode deixar comigo.", overlap=0.9),
    ],
}


def decode_mp3(path: Path) -> array.array:
    decoded = miniaudio.decode_file(
        str(path),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=SAMPLE_RATE,
    )
    samples = array.array("h")
    samples.frombytes(bytes(decoded.samples))
    return samples


def write_wav(path: Path, samples: array.array) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())


def extend_with_silence(track: array.array, length: int) -> None:
    if length > len(track):
        track.extend(array.array("h", [0] * (length - len(track))))


def mix_into(track: array.array, samples: array.array, start: int) -> int:
    end = start + len(samples)
    extend_with_silence(track, end)
    for offset, sample in enumerate(samples):
        mixed = track[start + offset] + sample
        track[start + offset] = max(MIN_SAMPLE, min(MAX_SAMPLE, mixed))
    return end


async def synthesize_mp3(text: str, voice: str, dest: Path) -> None:
    communicate = edge_tts.Communicate(text, voice=voice, rate="-5%")
    await communicate.save(str(dest))


async def synthesize_turns(
    turns: list[Turn], voices: tuple[str, str], tmp_dir: Path
) -> list[array.array]:
    async def one(index: int, turn: Turn) -> array.array:
        mp3_path = tmp_dir / f"turn_{index:02d}.mp3"
        await synthesize_mp3(turn.text, voices[turn.speaker], mp3_path)
        return decode_mp3(mp3_path)

    return list(await asyncio.gather(*[one(i, turn) for i, turn in enumerate(turns)]))


def mix_dialogue(turns: list[Turn], clips: list[array.array]) -> tuple[array.array, ...]:
    mixed = array.array("h")
    stems = (array.array("h"), array.array("h"))
    playhead = 0

    for turn, clip in zip(turns, clips, strict=True):
        if turn.overlap > 0:
            start = max(0, playhead - int(turn.overlap * SAMPLE_RATE))
        else:
            start = playhead + int(turn.gap * SAMPLE_RATE)
        end = mix_into(mixed, clip, start)
        mix_into(stems[turn.speaker], clip, start)
        playhead = max(playhead, end)

    for stem in stems:
        extend_with_silence(stem, len(mixed))
    return mixed, stems[0], stems[1]


async def generate(locale: str) -> None:
    voices = VOICES[locale]
    turns = DIALOGUES[locale]

    with tempfile.TemporaryDirectory() as tmp:
        clips = await synthesize_turns(turns, voices, Path(tmp))

    mixed, person1, person2 = mix_dialogue(turns, clips)
    dialogue_path = OUTPUT_DIR / "dialogue.wav"
    person1_path = OUTPUT_DIR / "person1.wav"
    person2_path = OUTPUT_DIR / "person2.wav"
    write_wav(dialogue_path, mixed)
    write_wav(person1_path, person1)
    write_wav(person2_path, person2)
    overlaps = sum(1 for turn in turns if turn.overlap > 0)
    duration = len(mixed) / SAMPLE_RATE
    print(f"Wrote {dialogue_path} ({duration:.1f}s, {overlaps} overlaps)")
    print(f"Wrote {person1_path}")
    print(f"Wrote {person2_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Portuguese two-speaker dialogue WAV."
    )
    parser.add_argument(
        "--locale",
        choices=sorted(VOICES),
        default="pt-BR",
        help="Portuguese locale (default: pt-BR).",
    )
    args = parser.parse_args()
    asyncio.run(generate(args.locale))


if __name__ == "__main__":
    main()
