# Changelog

All notable changes to this project will be documented
in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.1.0 2026-09-01 feature/initial

### Added

- OCR region selection and text extraction via an interactive
  click-and-drag overlay (`ocr-tts ocr`), powered by tkinter,
  `mss`/Pillow screen capture, and `pytesseract`.
- Multi-backend screen capture covering X11, Wayland compositors,
  nested compositors such as gamescope (PipeWire/GStreamer), and
  external tools (`grim`, `maim`, `scrot`, `import`, `spectacle`,
  `gnome-screenshot`), with an optional `OCR_TTS_CAPTURE_COMMAND`
  override and logical-to-capture resolution remapping.
- OCR options for language selection (`--lang`), custom tesseract
  path (`--tesseract-cmd`), saving the captured region
  (`--save-image`), and blank-frame detection warnings.
- Clipboard integration for copying extracted text or the captured
  region image to the system clipboard (wl-copy, xclip, xsel,
  pbcopy, clip.exe, PowerShell, osascript).
- Piper TTS engine for fully offline local text-to-speech synthesis
  on ONNX Runtime, with automatic voice-model download from the
  rhasspy/piper-voices HuggingFace repository.
- Friendly voice aliases (`male`, `female`) and per-utterance
  speed control (0.5–2.0).
- Streaming live player (`ocr-tts live`) that reads text from
  stdin and speaks each line as soon as its first audio chunk is
  synthesized, with continuous FIFO feeding.
- FastAPI TTS server (`ocr-tts api launch`) with non-streaming
  synthesis (`POST /synthesize`), output streaming
  (`POST /synthesize/stream`), input queueing (`POST /queue`),
  queued output streaming (`GET /queue/stream`), queue clearing
  (`POST /queue/clear`), graceful shutdown (`POST /shutdown`),
  voice listing (`GET /voices`), and voice downloading
  (`POST /download`).
- Queue-based TTS with per-item voice/speed, remote control via
  `ocr-tts api send-text` / `api send-region`, automatic server
  launch on connection-refused, a dedicated `api clear` command
  that wipes the queue and stops playback immediately, and
  `api close` for graceful server teardown.
- Verbose latency reporting (`--verbose` on `api send-text` /
  `api send-region`) with synthesis latency,
  piper-to-speech latency, turnaround time, and a per-stage
  processing breakdown for region selection.
- Global hotkey watcher (`ocr-tts hotkey-watcher start`) with
  pynput-based global hotkeys, YAML configuration, and actions:
  `speak-text`, `queue-clear`, `shutdown`, `ocr-region`,
  `send-region`, `launch`, and `sequence` (ordered sub-actions
  that stop at the first failure).
- Bundled example hotkey configuration (`hotkeys.example.yaml`).
- `scripts/ocr-region.sh` curl helper for driving the `/queue`
  endpoint from the shell.
- Uniform Typer CLI consolidating every entry point under a
  single `ocr-tts` command.
- Pydantic data validation with user-friendly CLI error messages.
- Pre-commit hooks via prek (ruff, mypy, pymarkdownlnt,
  shellcheck).
- `py.typed` marker for PEP 561 type stub compliance.
- CI matrix testing across Python 3.10–3.13, `pip-audit` for
  dependency vulnerability scanning, and strict mypy compliance.
