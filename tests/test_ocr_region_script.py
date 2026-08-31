"""Tests for the scripts/ocr-region.sh helper."""

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "ocr-region.sh"


@pytest.fixture(autouse=True)
def _fake_curl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a stub `curl` on PATH that prints its arguments."""
    curl = tmp_path / "curl"
    curl.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
    curl.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the helper via bash (robust regardless of the exec bit)."""
    return subprocess.run(  # noqa: S603
        ["bash", str(SCRIPT), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )


def test_speak_runs_curl_with_payload() -> None:
    """Speak POSTs the resolved voice/speed/text as /queue JSON."""
    result = run_script(
        "speak",
        "Hello world",
        "--voice",
        "male",
        "--speed",
        "1.3",
        "--host",
        "h",
        "--port",
        "9000",
    )
    assert result.returncode == 0
    assert "http://h:9000/queue" in result.stdout
    assert "--data" in result.stdout
    expected = json.dumps(
        {
            "text": "Hello world",
            "voice": "en_US-hfc_male-medium",
            "speed": 1.3,
        }
    )
    assert expected in result.stdout


def test_region_runs_curl_with_ocr_text() -> None:
    """Region posts the OCR'd text using the resolved alias."""
    result = run_script("region", "Extracted text", "--voice", "female")
    assert result.returncode == 0
    expected = json.dumps(
        {
            "text": "Extracted text",
            "voice": "en_US-hfc_female-medium",
            "speed": 1.0,
        }
    )
    assert expected in result.stdout


def test_payload_escapes_special_chars() -> None:
    """Double quotes and backslashes in the text are JSON-escaped."""
    text = 'say "hi" \\ now'
    result = run_script("speak", text, "--voice", "male")
    assert result.returncode == 0
    expected = json.dumps(
        {"text": text, "voice": "en_US-hfc_male-medium", "speed": 1.0}
    )
    assert expected in result.stdout


def test_speak_requires_text() -> None:
    """Speak without a text argument fails."""
    result = run_script("speak")
    assert result.returncode != 0
    assert "requires a <text>" in result.stderr


def test_unknown_option_fails() -> None:
    """An unknown flag is rejected."""
    result = run_script("speak", "hi", "--bogus", "x")
    assert result.returncode != 0
    assert "unknown option" in result.stderr
