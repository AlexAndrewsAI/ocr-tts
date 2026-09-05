"""Additional coverage tests for the text2speech module."""

import json
import wave
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import typer
from piper import AudioChunk
from typer.testing import CliRunner

from ocr_tts.text2speech import DEFAULT_VOICE, PiperTTS, app


@pytest.fixture
def voice_files(tmp_path: Path) -> Path:
    """Create dummy Piper voice files so no download is attempted."""
    (tmp_path / f"{DEFAULT_VOICE}.onnx").touch()
    config = tmp_path / f"{DEFAULT_VOICE}.onnx.json"
    config.write_text(json.dumps({"sample_rate": 22050}))
    return tmp_path


@pytest.fixture
def mock_voice() -> MagicMock:
    """Provide a mocked PiperVoice yielding a single AudioChunk."""
    voice = MagicMock()
    voice.config.sample_rate = 22050

    def _synthesize(
        _text: str,
        _syn_config: object | None = None,
    ) -> Iterator[AudioChunk]:
        yield _chunk()

    voice.synthesize.side_effect = _synthesize
    return voice


def _chunk(sample_rate: int = 22050) -> AudioChunk:
    """Build a minimal AudioChunk."""
    return AudioChunk(
        sample_rate=sample_rate,
        sample_width=2,
        sample_channels=1,
        audio_float_array=np.zeros(160, dtype=np.float32),
        phonemes=[],
        phoneme_ids=[],
    )


class TestSampleRateProperty:
    """Tests for the loaded-voice sample rate accessor."""

    def test_returns_loaded_voice_rate(
        self, mock_voice: MagicMock, voice_files: Path
    ) -> None:
        """With a loaded voice, its configured rate is returned."""
        with patch("ocr_tts.text2speech.PiperVoice") as mock_piper:
            mock_piper.load.return_value = mock_voice
            tts = PiperTTS(voice=DEFAULT_VOICE, voice_dir=str(voice_files))
            tts._load_voice()
        assert tts.sample_rate == 22050

    def test_defaults_without_voice(self, voice_files: Path) -> None:
        """Without a loaded voice the fallback rate is returned."""
        with patch("ocr_tts.text2speech.PiperVoice"):
            tts = PiperTTS(voice=DEFAULT_VOICE, voice_dir=str(voice_files))
        assert tts.sample_rate == 22050


class TestSynthesizeToWavEdgeCases:
    """Tests for synthesize_to_wav edge paths."""

    def test_no_audio_warns_and_returns(
        self, mock_voice: MagicMock, voice_files: Path, tmp_path: Path
    ) -> None:
        """An empty chunk stream produces no file and warns."""
        mock_voice.synthesize.side_effect = lambda *_a, **_k: iter([])
        with patch("ocr_tts.text2speech.PiperVoice") as mock_piper:
            mock_piper.load.return_value = mock_voice
            tts = PiperTTS(voice=DEFAULT_VOICE, voice_dir=str(voice_files))
            output = tmp_path / "out.wav"
            tts.synthesize_to_wav("Hello", str(output))
        assert not output.exists()

    def test_multiple_chunks_all_written(
        self, mock_voice: MagicMock, voice_files: Path, tmp_path: Path
    ) -> None:
        """Every streamed chunk lands in the WAV file."""
        chunks = [_chunk(), _chunk(), _chunk()]
        mock_voice.synthesize.side_effect = lambda *_a, **_k: iter(chunks)
        with patch("ocr_tts.text2speech.PiperVoice") as mock_piper:
            mock_piper.load.return_value = mock_voice
            tts = PiperTTS(voice=DEFAULT_VOICE, voice_dir=str(voice_files))
            output = tmp_path / "multi.wav"
            tts.synthesize_to_wav("Hello", str(output))
        assert output.exists()
        with wave.open(str(output), "rb") as wav:
            frames = wav.getnframes()
        # Three chunks of 160 float32 samples converted to int16 frames.
        assert frames == len(chunks) * 160


class TestMainExitPassthrough:
    """Tests for the CLI's typer.Exit passthrough."""

    def test_exit_from_synthesis_is_reraised(self, runner: CliRunner) -> None:
        """typer.Exit raised during synthesis propagates unchanged."""
        tts_instance = MagicMock()
        tts_instance.synthesize_to_wav.side_effect = typer.Exit(code=3)
        with patch("ocr_tts.text2speech.PiperTTS", return_value=tts_instance):
            result = runner.invoke(app, ["Hello world"])
        assert result.exit_code == 3
