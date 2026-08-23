#!/usr/bin/env python3
"""Record microphone audio and save it as a WAV file."""

from __future__ import annotations

import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

DEFAULT_SAMPLERATE = 16000
DEFAULT_CHANNELS = 1
RECORDINGS_DIR = Path("recordings")
# Virtual mixers follow the OS current input (built-in mic or headphones).
_OS_DEFAULT_NAMES = ("default", "pulse", "pipewire")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record audio from the current system microphone and save a WAV file."
    )
    parser.add_argument(
        "--duration",
        type=float,
        metavar="SECONDS",
        help="Record for a fixed number of seconds instead of waiting for Enter.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=DEFAULT_SAMPLERATE,
        help=f"Sample rate in Hz (default: {DEFAULT_SAMPLERATE}).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
        help=f"Number of input channels (default: {DEFAULT_CHANNELS}).",
    )
    parser.add_argument(
        "--device",
        help="Input device index or name. Defaults to the microphone currently selected in the OS.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List input devices and exit.",
    )
    return parser.parse_args()


def parse_device_arg(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def list_input_devices() -> None:
    devices = sd.query_devices()
    current = current_input_device()
    print("Input devices (* = current OS microphone):")
    for index, info in enumerate(devices):
        if info["max_input_channels"] < 1:
            continue
        marker = "*" if index == current else " "
        print(
            f"{marker} {index}: {info['name']} "
            f"({info['max_input_channels']} in, "
            f"{int(info['default_samplerate'])} Hz)"
        )


def current_input_device() -> int:
    """Resolve the microphone the OS is using right now."""
    devices = sd.query_devices()
    for name in _OS_DEFAULT_NAMES:
        for index, info in enumerate(devices):
            if info["max_input_channels"] < 1:
                continue
            if info["name"].strip().lower() == name:
                return index

    default_input = sd.default.device[0]
    if default_input is not None and int(default_input) >= 0:
        return int(default_input)

    for index, info in enumerate(devices):
        if info["max_input_channels"] >= 1:
            return index
    raise RuntimeError("No input device found.")


def resolve_input_device(explicit: str | None) -> int:
    if explicit is None:
        return current_input_device()
    info = sd.query_devices(parse_device_arg(explicit), "input")
    devices = sd.query_devices()
    for index, candidate in enumerate(devices):
        if (
            candidate["name"] == info["name"]
            and candidate["hostapi"] == info["hostapi"]
        ):
            return index
    raise RuntimeError(f"Input device not found: {explicit}")


def describe_device(index: int) -> str:
    info = sd.query_devices(index)
    return f"{info['name']} (index {index})"


def choose_input_settings(
    device: int, samplerate: int, channels: int
) -> tuple[int, int]:
    """Pick a capture rate/channel count the device will actually open."""
    info = sd.query_devices(device, "input")
    rates: list[int] = []
    for rate in (samplerate, int(info["default_samplerate"]), 48000, 44100):
        if rate > 0 and rate not in rates:
            rates.append(rate)

    max_channels = int(info["max_input_channels"])
    channel_options: list[int] = []
    for count in (channels, min(2, max_channels), max_channels):
        if 1 <= count <= max_channels and count not in channel_options:
            channel_options.append(count)

    last_error: Exception | None = None
    for rate in rates:
        for count in channel_options:
            try:
                sd.check_input_settings(
                    device=device, samplerate=rate, channels=count
                )
                return rate, count
            except Exception as exc:
                last_error = exc
    raise RuntimeError(
        f"Could not open {describe_device(device)} "
        f"at {samplerate} Hz / {channels} ch: {last_error}"
    )


def output_path() -> Path:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return RECORDINGS_DIR / f"{timestamp}.wav"


def wait_for_stop(duration: float | None) -> None:
    if duration is not None:
        if duration <= 0:
            raise ValueError("--duration must be greater than 0.")
        print(f"Recording for {duration:g} seconds... ", flush=True)
        threading.Event().wait(duration)
        return

    print("Recording... press Enter to stop.", flush=True)
    try:
        input()
    except EOFError:
        # Non-interactive stdin: keep recording until Ctrl+C.
        threading.Event().wait()


def record(
    samplerate: int,
    channels: int,
    duration: float | None,
    device: int,
) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []

    def callback(indata: np.ndarray, frames: int, time, status: sd.CallbackFlags) -> None:
        if status:
            print(status, file=sys.stderr)
        chunks.append(indata.copy())

    capture_rate, capture_channels = choose_input_settings(
        device, samplerate, channels
    )
    print(f"Using microphone: {describe_device(device)}", flush=True)
    if capture_rate != samplerate or capture_channels != channels:
        print(
            f"Device does not support {samplerate} Hz / {channels} ch; "
            f"capturing at {capture_rate} Hz / {capture_channels} ch.",
            flush=True,
        )
    try:
        with sd.InputStream(
            device=device,
            samplerate=capture_rate,
            channels=capture_channels,
            callback=callback,
        ):
            wait_for_stop(duration)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)

    if not chunks:
        raise RuntimeError("No audio was captured.")
    return np.concatenate(chunks, axis=0), capture_rate


def main() -> int:
    args = parse_args()
    if args.list_devices:
        list_input_devices()
        return 0

    try:
        device = resolve_input_device(args.device)
        audio, samplerate = record(
            args.samplerate, args.channels, args.duration, device
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    path = output_path()
    sf.write(path, audio, samplerate)
    print(f"Saved {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
