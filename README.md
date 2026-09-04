# ocr-tts

OCR and TTS toolkit for screen region text extraction.

## Overview

| Purpose            | Tool      |
| ------------------ | --------- |
| Package Management | uv        |
| Data Validation    | pydantic  |
| CLI Framework      | typer     |
| OCR Engine         | tesseract |
| Screen Capture     | mss       |
| Image Processing   | Pillow    |
| Testing            | pytest    |
| Code Quality       | ruff      |
| Type Checking      | mypy      |
| Security Audit     | pip-audit |
| Markdown Lint      | pymarkdownlnt |
| Audio Playback     | sounddevice |
| Git Hooks          | prek      |

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  installed on your system
- [PortAudio](http://www.portaudio.com/) (for sounddevice playback)

### Installing Tesseract

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Installing PortAudio (for sounddevice)

```bash
# Ubuntu/Debian
sudo apt install libportaudio2

# macOS
brew install portaudio
```

## Installation

```bash
cd ocr-tts
uv sync --dev
```

To install the package in editable mode:

```bash
uv pip install -e .
ocr-tts --version
```

## Usage

Every entry point is a subcommand of the single `ocr-tts` CLI:

```bash
uv run ocr-tts --help
```

```text
ocr-tts
├── ocr           Select a region and extract text via OCR
├── text2speech   Convert text to speech (WAV)
├── api
│   ├── launch        Run the FastAPI server
│   ├── send-text     Add text to the running server queue
│   ├── send-region   Select a region, OCR it, and queue the text
│   ├── clear         Wipe the queue and immediately stop playback
│   └── close         Tear down the running server and its subprocesses
└── live          Stream text from stdin and play it back
```

### Command Line Interface

#### OCR Region Extraction

```bash
# Show version
uv run ocr-tts --version

# Select a region and extract text (default English)
uv run ocr-tts ocr

# Use a different language
uv run ocr-tts ocr --lang fra

# Specify custom tesseract path
uv run ocr-tts ocr --tesseract-cmd /usr/bin/tesseract
```

#### Text to Speech

```bash
# Convert text to speech (auto-downloads voice models)
uv run ocr-tts text2speech "Hello, world!"

# Specify output file
uv run ocr-tts text2speech "Hello!" -o greeting.wav

# Use a different Piper voice (default: en_US-hfc_male-medium)
uv run ocr-tts text2speech "Bonjour!" -v fr_FR-siwis-medium -o french.wav

# Adjust speed (0.5-2.0)
uv run ocr-tts text2speech "Hello!" --speed 0.8
```

##### Streaming Player (Live Mode)

Stream text immediately as it is spoken, with support for appending
new text at any time:

```bash
# Read text line-by-line from stdin and speak immediately
uv run ocr-tts live --speed 1.0

# Test with a simple example
echo "Hello, this is being spoken immediately" | uv run ocr-tts live

# Use a different voice
echo "Bonjour!" | uv run ocr-tts live -v fr_FR-siwis-medium

# Use a faster speed
cat message.txt | uv run ocr-tts live --speed 1.5
```

The live player:

- Starts playback as soon as the first audio chunk is generated (streaming)
- Accepts new lines at any time (continuous feed)
- Plays all queued text in FIFO order

##### Remote Queue Control (`api send-text`)

Add text to the running server's queue:

```bash
# Add text to the queue (works even when the queue is empty)
uv run ocr-tts api send-text "Hello, world!"

# Use a different voice and speed for this item
uv run ocr-tts api send-text "Bonjour!" -v fr_FR-siwis-medium -s 1.2

# Wipe the queue and immediately stop playback
uv run ocr-tts api clear
```

Each queued item carries its own voice and speed.  Switching voice/speed
mid-queue only affects text submitted after the switch; text already in
the queue keeps the settings it was submitted with.

###### Verbose latency reporting (`--verbose`)

Pass `--verbose` (also available on `api send-region`) to make the
command block the `POST /queue` until the audio for the submitted text
has **started playing**, then report the latency the API measures plus
the total turnaround time — the wall-clock time from when the process was
launched (captured at package import, before the heavy TTS imports)
until the API returned the latency:

```bash
uv run ocr-tts api send-text "Hello, world!" --verbose
```

Example output:

```text
Queued: 1 item(s) pending in the queue
Latency: synthesis=1924.997 ms, piper-to-speech=340.5 ms
turnaround-time: 3.231 s
```

`synthesis` is the time the server spent generating the audio; `piper-to-speech`
is the time from when the model started until the first audio chunk was
spoken (reported as `n/a` when no audio device is available, e.g. a
headless server). `turnaround-time` includes the client's Python startup
and heavy imports (Piper/ONNX Runtime, FastAPI, sounddevice) in addition
to the API latency, so it is always greater than or equal to the
reported latencies.

###### `send-region --verbose` breakdown

`api send-region` accepts the same `--verbose` flag, but because it is
interactive it prints a per-stage breakdown and adjusts the turnaround
time to exclude the user's click-drag time:

```bash
uv run ocr-tts api send-region --verbose
```

Example output:

```text
region-ui-load: 0.412 s
user-region-select: 1.873 s
capture: 0.038 s
ocr: 0.215 s
Latency: synthesis=463.427 ms, piper-to-speech=856.48 ms
turnaround-time: 2.231 s
```

The breakdown stages are:

- `region-ui-load` — time to load the selection overlay, up to the
  handoff to the user.
- `user-region-select` — time the user spent click-dragging to select
  the region. Shown for visibility but **not** counted in
  `turnaround-time`.
- `capture` — time to grab the selected screen region.
- `ocr` — time for Tesseract to extract text from the captured image.

`turnaround-time` for `send-region` therefore measures from process
launch through the API returning the latency, **minus** the user's
interactive `user-region-select` time — i.e. it reflects the client's
processing cost (startup + region UI load + capture + OCR + API
synthesis) rather than how long you took to draw the box.

##### Shutting Down the Server (`api close`)

`send-text` and `clear` auto-launch the server in the background when it is not
already running.  To tear that server (and any subprocesses it spawned)
back down:

```bash
# Stop the background server on the default port
uv run ocr-tts api close

# Stop a server running on a custom port
uv run ocr-tts api close --port 9000
```

`close` sends a `POST /shutdown` command to the running server.  The server
then tears itself down internally: it stops the queue processor and
playback thread, drains pending work, and exits the uvicorn process
gracefully (including any subprocesses it started).  As a safety net, any
server process that does not exit after the command is sent is
force-terminated.  If no server is running it is a no-op.

##### Curl Helper (`scripts/ocr-region.sh`)

A small bash helper that POSTs text to the `/queue` endpoint with curl — the
same request `api send-text` / `api send-region` make — for driving the server
from the shell:

```bash
# Queue spoken text (equivalent to `api send-text`)
scripts/ocr-region.sh speak "Hello world" --voice male --speed 1.3

# Queue text that was OCR'd from a screen selection
# (equivalent to `api send-region`)
scripts/ocr-region.sh region "extracted text" --voice female
```

Both default to `--voice male --speed 1.0 --host 127.0.0.1 --port 8000`; use
`--voice`, `--speed`, `--host`, and `--port` to override.

### API Endpoints

The package includes a FastAPI server with streaming text-to-speech,
supporting both input queueing and output streaming:

```bash
# Start the API server (http://localhost:8000)
uv run ocr-tts api launch

# Synthesize to a WAV file
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' --output hello.wav

# Stream raw PCM audio as it is generated
curl -N -X POST http://localhost:8000/synthesize/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' --output hello.raw

# Queue text to be synthesized (input streaming)
curl -X POST http://localhost:8000/queue \
  -H "Content-Type: application/json" \
  -d '{"text": "First sentence."}'

# Stream audio from the queue (output streaming)
curl -N http://localhost:8000/queue/stream --output queue.raw

# Wipe the queue and immediately stop playback
curl -X POST http://localhost:8000/queue/clear
```

See [TEXT2SPEECH.md](TEXT2SPEECH.md) for detailed documentation.

### Global Hotkey Watcher

A background service that watches for hotkey presses and dispatches
OCR-TTS actions (speak-text, clear queue, shutdown, OCR region
selection, send-region, launch server, or sequences of actions).

```bash
# Start the hotkey watcher with the bundled example config
uv run ocr-tts hotkey-watcher start

# Use a custom configuration file
uv run ocr-tts hotkey-watcher start --config ~/.config/ocr-tts/hotkeys.yaml
```

The watcher can be configured with a YAML file (see
[hotkeys.example.yaml](hotkeys.example.yaml) for details):

```yaml
hotkeys:
  - hotkey: <ctrl>+<shift>+s
    action: speak-text
    text: Hello world!
    voice: en_US-hfc_male-medium
    speed: 1.0
```

The bundled [hotkeys.example.yaml](hotkeys.example.yaml) provides examples for:

- `speak-text` — speak queued text
- `queue-clear` — wipe the queue and stop playback
- `shutdown` — tear down the running server and its subprocesses
- `ocr-region` — select a region, OCR it, and queue the text
- `send-region` — queue text from a screen selection
- `launch` — start the API server if not running
- `sequence` — run a sequence of actions, stopping at the first failure

## Development

### Install Dev Dependencies

```bash
uv sync --dev
```

### Run Tests

```bash
# Run all tests
uv run pytest

# Run only the streaming player tests
uv run pytest tests/test_player.py

# Run tests with coverage report
uv run pytest --cov=ocr_tts

# Run tests targeting the player module
uv run pytest --cov=ocr_tts ocr_tts/player.py

# Generate coverage report as HTML
uv run pytest --cov=ocr_tts --cov-report=html
```

### Code Quality

```bash
uv run ruff check .
uv run ruff format .
uv run mypy .
uv run pip-audit
```

### Git Hooks with prek

```bash
uv run prek install
```

## Project Structure

```text
ocr-tts/
├── AGENTS.md
├── CHANGELOG.md
├── .gitignore
├── pyproject.toml
├── README.md
├── TEXT2SPEACH.md
├── ocr_tts
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── ocr_region.py
│   ├── text2speech.py
│   ├── api.py
│   ├── player.py
│   ├── queue.py
├── tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_text2speech.py
│   ├── test_api.py
│   ├── test_player.py
│   ├── test_queue.py
└── uv.lock
```

## License

This project is licensed under the
[GNU Affero General Public License v3](LICENSE).
