# text2speech

Text-to-speech wrapper for [piper-tts](https://github.com/rhasspy/piper) that
converts text to speech and saves it as a WAV file, with a streaming FastAPI
server that supports both input queueing and output streaming.

## Installation

The piper-tts dependency will be automatically installed when you install
ocr-tts:

```bash
uv sync --dev
```

## CLI Usage

### Basic Usage

Convert text to speech with default settings:

```bash
text2speech "Hello, world!"
```

This will:

- Use the default voice (`en_US-hfc_male-medium`)
- Save output to `output.wav` in the current directory
- Automatically download the required model files to `.piper-voices/` (if missing)

### Specify Output File

```bash
text2speech "Hello, world!" -o greeting.wav
text2speech "Hello, world!" --output greeting.wav
```

### Choose a Voice

```bash
text2speech "Hello!" -v en_US-lessac-medium
text2speech "Bonjour!" -v fr_FR-siwis-medium
```

### Adjust Speed

```bash
text2speech "Hello!" --speed 0.8  # Slower
text2speech "Hello!" --speed 1.5  # Faster
```

### Custom Voice Directory

```bash
text2speech "Hello!" --voice-dir /path/to/voices
```

## Available Voices

Piper supports many voices across different languages. Voice names follow the
pattern `<lang>_<REGION>-<name>-<quality>`, for example:

- `en_US-hfc_male-medium` - American male (default)
- `en_US-lessac-medium` - American male
- `en_US-libritts-high` - American male, high quality
- `en_GB-alba-medium` - British female
- `fr_FR-siwis-medium` - French female
- `de_DE-thorsten-medium` - German male

To download additional voices:

```bash
curl -X POST http://localhost:8000/download \
  -H "Content-Type: application/json" \
  -d '{"voice": "en_US-lessac-medium"}'
```

To list downloaded voices:

```bash
curl http://localhost:8000/voices
```

## Voice Model Storage

Voice models are stored in `.piper-voices/` by default (gitignored). This
directory contains, for each voice:

- `<voice>.onnx` - The Piper ONNX model
- `<voice>.onnx.json` - The voice configuration

These files are downloaded automatically on first use from the
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)
HuggingFace repository.

## Streaming API

The package includes a FastAPI server (`tts-api`) that supports both input
queueing and output streaming:

### Start the Server

```bash
uv run tts-api   # serves on http://localhost:8000
```

### Non-streaming synthesis (WAV)

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' --output hello.wav
```

### Output streaming (raw PCM)

Audio chunks are streamed as each sentence is processed, so playback can
start before the full text has been synthesized:

```bash
curl -N -X POST http://localhost:8000/synthesize/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' --output hello.raw
```

The response media type is `audio/L16; rate=22050; channels=1` (16-bit PCM).

### Input queueing + output streaming

Queue text with separate calls, then stream the audio output:

```bash
# Add text segments to the queue (input streaming)
curl -X POST http://localhost:8000/queue \
  -H "Content-Type: application/json" \
  -d '{"text": "First sentence."}'
curl -X POST http://localhost:8000/queue \
  -H "Content-Type: application/json" \
  -d '{"text": "Second sentence."}'

# Stream the synthesized audio as it becomes available (output streaming)
curl -N http://localhost:8000/queue/stream --output queue.raw
```

### Controlling the queue from the CLI (`speak`)

The `speak` command is a remote control for the running server's queue.
Text is appended to the queue and spoken after anything already queued,
so it works even when the queue is empty:

```bash
# Add text to the running queue (server must be running)
uv run speak "Hello, world!"

# Add text with a different voice and speed
uv run speak "Bonjour!" -v fr_FR-siwis-medium -s 1.2

# Point at a server on a different host/port
uv run speak "Hi" --host 192.168.1.10 --port 9000

# Wipe the queue and immediately stop playback
uv run ocr-tts api clear
```

Each item carries its own voice and speed.  Switching voice/speed
mid-queue affects only text submitted after the switch — text already in
the queue keeps the settings it was submitted with:

```bash
uv run speak "English text" -v en_US-lessac-medium
# still speaking in English...
uv run speak "Français" -v fr_FR-siwis-medium   # switches once reached
```

To clear the queue over HTTP directly, use `POST /queue/clear`; it wipes
all pending text and audio and stops playback of any in-flight item.

## Options

- `TEXT` - Required text to convert to speech
- `-o, --output PATH` - Output WAV file path (default: output.wav)
- `-v, --voice NAME` - Voice name (default: en_US-hfc_male-medium)
- `--voice-dir PATH` - Directory for voice/model files (default: .piper-voices)
- `-s, --speed FLOAT` - Speech speed multiplier, 0.5-2.0 (default: 1.0)
- `-V, --version` - Show version and exit

## Examples

### Simple greeting

```bash
text2speech "Welcome to OCR-TTS toolkit!"
```

### French text

```bash
text2speech "Bonjour, comment allez-vous?" -v fr_FR-siwis-medium -o french.wav
```

### Slow, clear speech

```bash
text2speech "This is a test of the text-to-speech system." --speed 0.8 -o slow.wav
```

### Fast speech

```bash
text2speech "Quick brown fox jumps over the lazy dog." --speed 1.5 -o fast.wav
```

## Error Handling

The CLI will automatically download missing model files. If download fails, an
error message will be displayed with instructions.

Common issues:

- **Model files missing**: Will be auto-downloaded on first run
- **Network errors**: Check internet connection for model download
- **Invalid voice name**: Use a valid voice name matching
  `en_US-hfc_male-medium` style
- **Permission errors**: Ensure write access to output directory and voice
  directory

## Technical Details

- Uses [piper-tts](https://github.com/rhasspy/piper) backend
- Built on [ONNX Runtime](https://onnxruntime.ai/) — runs fully offline after
  model download
- Output format: WAV (16-bit PCM), mono
- Sample rate: 22050 Hz (voice default)
- Model size: ~60MB per voice (downloaded once)
- Streaming: `PiperVoice.synthesize()` yields one `AudioChunk` per sentence

## Related

- [piper-tts GitHub](https://github.com/rhasspy/piper)
- [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)
- [ocr-region CLI](../README.md) - OCR screen region selection
