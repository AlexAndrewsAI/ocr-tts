# Agent Instructions: ocr-tts

## Quick Start

1. **Setup:** Run `uv sync --dev` before major work sessions
2. **Activate:** Ensure `.venv` is active; run `uv venv` if missing
3. **Code:** Use `python3` or `uv run python`; always add type hints and tests

## Tech Stack

| Component            | Tool      |
| -------------------- | --------- |
| Environment & Deps   | uv        |
| Data Validation      | Pydantic  |
| CLI Framework        | Typer     |
| OCR Engine           | Tesseract / pytesseract |
| Screen Capture       | mss       |
| Image Processing     | Pillow    |
| Testing              | pytest    |
| Linting & Formatting | ruff      |
| Type Checking        | mypy      |
| Security Audit       | pip-audit |
| Markdown Lint        | pymarkdownlnt |
| Git Hooks            | prek      |

## Project Structure

```text
ocr_tts/
  __init__.py         (Package init)
  __main__.py         (Module entry point)
  cli.py              (Typer CLI)
  ocr_region.py       (OCR region selection & extraction)
  desktop.py          (Platform services: screen capture backends + clipboard)
  text2speech.py      (Piper TTS engine + CLI)
  api.py              (FastAPI server: synthesize + queue endpoints)
  player.py           (Live streaming playback)
  queue.py            (speak CLI: remote control of the server queue)
  hotkey_watcher.py   (Global hotkey service dispatching OCR-TTS actions)
tests/                (Pytest suite)
pyproject.toml        (Dependencies & tool config)
```

## Essential Directives

### Code Standards

- **Type Hints:** Required on ALL function signatures
  and class members. Enforce strictly with mypy.
  Avoid using `# type: ignore` comments to suppress
  mypy errors; fix the underlying type issues instead.
- **Docstrings:** Google-style format for all public APIs.
- **Logging:** Use `logging` module only; never `print()`.
- **Relative Paths:** Never use absolute paths in code.

### Dependency & Configuration Management

- **Adding/Removing Dependencies:** Use `uv add` /
  `uv remove` commands.
- **Editing pyproject.toml:** Avoid manual edits during
  development. Only update `pyproject.toml` as the
  **final change** after all work is tested.
- **Before Major Work:** Always run `uv sync --dev` first.

### Testing & Quality

- **Test Coverage:** Every code change requires
  corresponding tests in `tests/`.
- **Validation Before Commit:** Run the full suite:
  `uv run pytest`, `uv run ruff check .`,
  `uv run mypy .`, `uv run pip-audit`.
- **Pre-commit Hooks:** Use `uv run prek install` to
  set up Git hooks that automatically run checks.

### Operational Constraints

- **No Interactive Prompts:** Mock or bypass any
  interactive commands.
- **Staging & Commit Protocol:** When you have completed
  work and updated files, stage the changes with
  `git add` and then display a suggested commit message
  for the user's review. DO NOT actually commit.
- **Code Review Mode:** Analyze only; record findings
  in `./REVIEW.md` without making modifications.

### File Maintenance

- **Keep Instructions Current:** Update "Tech Stack,"
  "Project Structure," and "Workflow Commands" if
  `pyproject.toml`, structure, or core logic changes.
- **Pre-commit Config:** Keep `.pre-commit-config.yaml`
  in sync with CI workflow when test requirements change.
- **YAML Files:** Use `.yaml` extension for all YAML files
  (e.g., `.github/workflows/ci.yaml` instead of `.yml`).

## Workflow Commands

```bash
uv sync --dev                    # Install/sync deps
uv run pytest                    # Run tests
uv run ruff check .              # Lint
uv run ruff format .             # Auto-format
uv run mypy .                    # Type check
uv run pip-audit                 # Security audit
uv run prek install              # Install git hooks
uv run prek run --all-files      # Run all hooks
uv run ocr-tts --version          # Test CLI
uv run ocr-tts api send-text --help  # Test queue remote-control CLI
uv run ocr-tts api clear --help      # Test the queue clear subcommand
```
