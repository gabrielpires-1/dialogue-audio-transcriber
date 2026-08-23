# dialogue-audio-transcriber

Record microphone audio from the command line and save it as a WAV file.

## Setup

On Linux, install PortAudio first (required by `sounddevice`):

```bash
sudo apt install libportaudio2
```

Then install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Record from the microphone currently selected in the OS (built-in or headphones) until you press Enter (or Ctrl+C):

```bash
python record.py
```

The script prints the device it opened, for example `Using microphone: default (index 3)`.

Record for a fixed number of seconds:

```bash
python record.py --duration 10
```

List input devices (the current OS microphone is marked with `*`):

```bash
python record.py --list-devices
```

Pick a device explicitly by index or name if you need to override:

```bash
python record.py --device 1
python record.py --device "Headset"
```

Optional flags:

- `--samplerate` — sample rate in Hz (default: 16000)
- `--channels` — input channels (default: 1)

Files are written to `recordings/YYYYMMDD-HHMMSS.wav`.
