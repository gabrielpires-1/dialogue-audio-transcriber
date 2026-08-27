from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from faster_whisper.audio import decode_audio
from pyannote.audio import Pipeline

from dialogue_audio_transcriber.constants import (
    ALLOWED_DIARIZATION_MODES,
    AUDIO_SAMPLE_RATE,
    DIARIZATION_API,
    INVALID_DIARIZATION_MESSAGE,
    MISSING_PYANNOTE_API_KEY_MESSAGE,
)
from dialogue_audio_transcriber.exceptions import (
    InvalidDiarizationModeError,
    MissingPyannoteApiKeyError,
)
from dialogue_audio_transcriber.speaker_turns import SpeakerTurn

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
API_DIARIZATION_MODEL = "pyannote/speaker-diarization-precision-2"
NUM_SPEAKERS = 2


class Diarizer(Protocol):
    def diarize(self, audio_path: Path) -> list[SpeakerTurn]: ...


def resolve_diarization_mode(diarization: str, api_key: str) -> str:
    if diarization not in ALLOWED_DIARIZATION_MODES:
        raise InvalidDiarizationModeError(INVALID_DIARIZATION_MESSAGE)
    if diarization == DIARIZATION_API and not (api_key or "").strip():
        raise MissingPyannoteApiKeyError(MISSING_PYANNOTE_API_KEY_MESSAGE)
    return diarization


def load_diarization_pipeline(hf_token: str, device: str) -> Pipeline:
    token = (hf_token or "").strip()
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set TRANSCRIBER_HF_TOKEN or HF_TOKEN "
            "and accept the pyannote/speaker-diarization-community-1 license."
        )
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    if pipeline is None:
        raise RuntimeError(f"Failed to load diarization pipeline {DIARIZATION_MODEL}.")
    if hasattr(pipeline, "segmentation_batch_size"):
        pipeline.segmentation_batch_size = 1
    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = 1
    pipeline.to(torch.device(device))
    return pipeline


def load_api_diarization_pipeline(api_key: str) -> Pipeline:
    token = (api_key or "").strip()
    if not token:
        raise RuntimeError(
            "Missing pyannoteAI API key. Set TRANSCRIBER_PYANNOTE_API_KEY or "
            "PYANNOTEAI_API_KEY to use API diarization."
        )
    os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
    pipeline = Pipeline.from_pretrained(API_DIARIZATION_MODEL, token=token)
    if pipeline is None:
        raise RuntimeError(
            f"Failed to load diarization pipeline {API_DIARIZATION_MODEL}."
        )
    return pipeline


def release_gpu_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _has_tracks(annotation: object) -> bool:
    if annotation is None:
        return False
    try:
        return len(annotation) > 0
    except TypeError:
        return True


def _annotation_from_output(output: object) -> object:
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    if _has_tracks(exclusive):
        return exclusive
    diarization = getattr(output, "speaker_diarization", None)
    if _has_tracks(diarization):
        return diarization
    if exclusive is not None:
        return exclusive
    if diarization is not None:
        return diarization
    return output


def _turns_from_annotation(annotation: object) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                SpeakerTurn(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker_id=str(speaker),
                )
            )
        return turns

    for item in annotation:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        turn, speaker = item
        turns.append(
            SpeakerTurn(
                start=float(turn.start),
                end=float(turn.end),
                speaker_id=str(speaker),
            )
        )
    return turns


def diarize_waveform(pipeline: Pipeline, waveform: np.ndarray) -> list[SpeakerTurn]:
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    elif audio.ndim > 1:
        audio = np.mean(audio, axis=0, keepdims=True)
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(audio))
        output = pipeline(
            {
                "waveform": tensor,
                "sample_rate": AUDIO_SAMPLE_RATE,
            },
            num_speakers=NUM_SPEAKERS,
        )
    return _turns_from_annotation(_annotation_from_output(output))


class PyannoteDiarizer:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        audio = decode_audio(str(audio_path), sampling_rate=AUDIO_SAMPLE_RATE)
        waveform = np.asarray(audio, dtype=np.float32)
        return diarize_waveform(self._pipeline, waveform)


class PyannoteApiDiarizer:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
        output = self._pipeline.apply(
            {"audio": str(audio_path)},
            num_speakers=NUM_SPEAKERS,
        )
        return _turns_from_annotation(_annotation_from_output(output))


def load_diarizer(hf_token: str, device: str) -> PyannoteDiarizer:
    return PyannoteDiarizer(load_diarization_pipeline(hf_token, device))


def load_api_diarizer(api_key: str) -> PyannoteApiDiarizer:
    return PyannoteApiDiarizer(load_api_diarization_pipeline(api_key))
