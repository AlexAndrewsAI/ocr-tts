"""Tests for the Piper-based text2speech module."""

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import typer
from piper import AudioChunk, SynthesisConfig
from typer.testing import CliRunner

from ocr_tts.text2speech import (
    DEFAULT_VOICE,
    PiperTTS,
    VoiceDownloadError,
    app,
    download_file,
    ensure_voice,
    get_voice_dir,
    get_voice_urls,
    parse_voice_name,
    resolve_voice_alias,
)


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
        _syn_config: SynthesisConfig | None = None,
    ) -> Iterator[AudioChunk]:
        yield AudioChunk(
            sample_rate=22050,
            sample_width=2,
            sample_channels=1,
            audio_float_array=np.zeros(1600, dtype=np.float32),
            phonemes=[],
            phoneme_ids=[],
        )

    voice.synthesize.side_effect = _synthesize
    return voice


class TestParseVoiceName:
    """Tests for parse_voice_name."""

    def test_valid_name(self) -> None:
        """Test parsing a valid Piper voice name."""
        result = parse_voice_name("en_US-hfc_male-medium")
        assert result == {
            "lang_family": "en",
            "lang_code": "en_US",
            "voice_name": "hfc_male",
            "voice_quality": "medium",
        }

    def test_invalid_name(self) -> None:
        """Test that an invalid voice name raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            parse_voice_name("not-a-valid-name")


class TestGetVoiceUrls:
    """Tests for get_voice_urls."""

    def test_returns_model_and_config_urls(self) -> None:
        """Test the HuggingFace download URLs are constructed correctly."""
        model_url, config_url = get_voice_urls("en_US-hfc_male-medium")
        assert "en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx" in model_url
        assert ".onnx.json" in config_url
        assert "?download=true" in model_url


class TestGetVoiceDir:
    """Tests for get_voice_dir."""

    def test_default_voice_dir(self) -> None:
        """Test default voice directory is .piper-voices."""
        with patch("ocr_tts.text2speech.Path") as mock_path:
            result = get_voice_dir()
            mock_path.assert_called_once_with(".piper-voices")
            assert result is mock_path.return_value
            result.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_custom_voice_dir(self, tmp_path: Path) -> None:
        """Test custom voice directory."""
        custom_dir = str(tmp_path / "custom-voices")
        with patch("ocr_tts.text2speech.Path") as mock_path:
            result = get_voice_dir(custom_dir)
            mock_path.assert_called_once_with(custom_dir)
            assert result is mock_path.return_value
            result.mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestEnsureVoice:
    """Tests for ensure_voice."""

    @patch("ocr_tts.text2speech.download_file")
    def test_downloads_missing_files(
        self, mock_download: MagicMock, tmp_path: Path
    ) -> None:
        """Test that missing model/config files are downloaded."""
        model_path, config_path = ensure_voice(DEFAULT_VOICE, tmp_path)
        assert mock_download.call_count == 2
        assert model_path == tmp_path / f"{DEFAULT_VOICE}.onnx"
        assert config_path == tmp_path / f"{DEFAULT_VOICE}.onnx.json"

    @patch("ocr_tts.text2speech.download_file")
    def test_skips_download_when_files_exist(
        self, mock_download: MagicMock, voice_files: Path
    ) -> None:
        """Test that existing model/config files are not downloaded."""
        model_path, config_path = ensure_voice(DEFAULT_VOICE, voice_files)
        mock_download.assert_not_called()
        assert model_path == voice_files / f"{DEFAULT_VOICE}.onnx"
        assert config_path == voice_files / f"{DEFAULT_VOICE}.onnx.json"


class TestDownloadFile:
    """Tests for download_file."""

    def test_invalid_url_scheme_rejected(self, tmp_path: Path) -> None:
        """Test that non-http(s) URL schemes are rejected."""
        dest = tmp_path / "test.txt"
        with pytest.raises(VoiceDownloadError) as exc_info:
            download_file("file:///etc/passwd", dest)
        assert "Invalid URL scheme" in str(exc_info.value)

    @patch("urllib.request.urlopen")
    def test_download_success(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        """Test successful file download writes the payload atomically."""
        dest = tmp_path / "test.txt"
        mock_urlopen.return_value.__enter__.return_value.read.side_effect = [
            b"hello ",
            b"world",
            b"",
        ]
        download_file("https://example.com/file.txt", dest)
        assert dest.read_bytes() == b"hello world"
        # No leftover partial/temp files remain.
        assert list(tmp_path.iterdir()) == [dest]

    @patch("urllib.request.urlopen")
    def test_download_failure(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        """Test download failure handling and temp cleanup."""
        dest = tmp_path / "test.txt"
        mock_urlopen.return_value.__enter__.return_value.read.side_effect = Exception(
            "Network error"
        )
        with pytest.raises(VoiceDownloadError) as exc_info:
            download_file("https://example.com/file.txt", dest)
        assert "Network error" in str(exc_info.value)
        # No partial or destination file is left behind.
        assert not dest.exists()
        assert list(tmp_path.iterdir()) == []

    def test_truncated_partial_download_is_cleaned_up(self, tmp_path: Path) -> None:
        """A failure mid-download removes the partial file (M10)."""
        dest = tmp_path / "test.bin"
        resp = MagicMock()
        resp.read.side_effect = [b"partial", Exception("connection reset")]

        class _Resp:
            def __enter__(self) -> MagicMock:
                return resp

            def __exit__(self, *args: object) -> None:
                return None

        with (
            patch("urllib.request.urlopen", return_value=_Resp()),
            pytest.raises(VoiceDownloadError),
        ):
            download_file("https://example.com/test.bin", dest)
        assert not dest.exists()
        # Temp files are removed, leaving an empty directory.
        assert list(tmp_path.iterdir()) == []


class TestPiperTTS:
    """Tests for the PiperTTS class."""

    @patch("ocr_tts.text2speech.PiperVoice")
    def test_synthesize_returns_chunks(
        self,
        mock_piper_voice: MagicMock,
        mock_voice: MagicMock,
        voice_files: Path,
    ) -> None:
        """Test that synthesize yields audio chunks."""
        mock_piper_voice.load.return_value = mock_voice
        tts = PiperTTS(voice=DEFAULT_VOICE, voice_dir=str(voice_files))
        chunks = list(tts.synthesize("Hello world"))
        assert len(chunks) == 1
        assert chunks[0].sample_rate == 22050
        mock_voice.synthesize.assert_called_once()

    @patch("ocr_tts.text2speech.PiperVoice")
    def test_synthesize_to_wav_writes_file(
        self,
        mock_piper_voice: MagicMock,
        mock_voice: MagicMock,
        voice_files: Path,
        tmp_path: Path,
    ) -> None:
        """Test that synthesize_to_wav writes a valid WAV file."""
        mock_piper_voice.load.return_value = mock_voice
        tts = PiperTTS(voice=DEFAULT_VOICE, voice_dir=str(voice_files))
        output = tmp_path / "out.wav"
        tts.synthesize_to_wav("Hello", str(output))
        assert output.exists()
        assert output.stat().st_size > 0

    @patch("ocr_tts.text2speech.PiperVoice")
    def test_sample_rate_property(
        self,
        mock_piper_voice: MagicMock,
        mock_voice: MagicMock,
        voice_files: Path,
    ) -> None:
        """Test the sample_rate property reflects the loaded voice."""
        mock_piper_voice.load.return_value = mock_voice
        tts = PiperTTS(voice=DEFAULT_VOICE, voice_dir=str(voice_files))
        tts.synthesize("Hi")
        assert tts.sample_rate == 22050


class TestText2speechCLI:
    """Tests for the text2speech CLI."""

    def test_app_exists(self) -> None:
        """Test that app is a Typer instance."""
        assert isinstance(app, typer.Typer)

    def test_version_callback(self, runner: CliRunner) -> None:
        """Test version flag."""
        result = runner.invoke(app, ["Hello", "--version"])
        assert result.exit_code == 0
        assert "text2speech" in result.output

    @patch("ocr_tts.text2speech.PiperTTS")
    def test_main_success(
        self,
        mock_tts: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test successful text-to-speech conversion."""
        tts_instance = MagicMock()
        mock_tts.return_value = tts_instance
        output = tmp_path / "output.wav"
        result = runner.invoke(app, ["Hello world", "-o", str(output)])
        assert result.exit_code == 0
        mock_tts.assert_called_once_with(voice=DEFAULT_VOICE, voice_dir=None)
        tts_instance.synthesize_to_wav.assert_called_once_with(
            "Hello world", str(output), 1.0
        )

    @patch("ocr_tts.text2speech.PiperTTS")
    def test_main_error_handling(self, mock_tts: MagicMock, runner: CliRunner) -> None:
        """Test error handling in main."""
        mock_tts.side_effect = Exception("Model load failed")
        result = runner.invoke(app, ["Hello world"])
        assert result.exit_code == 1
        assert "Error generating speech" in result.output

    @patch("ocr_tts.text2speech.PiperTTS")
    def test_main_voice_download_error(
        self, mock_tts: MagicMock, runner: CliRunner
    ) -> None:
        """A VoiceDownloadError is reported as a CLI error (exit 1)."""
        mock_tts.side_effect = VoiceDownloadError("Network error")
        result = runner.invoke(app, ["Hello world"])
        assert result.exit_code == 1
        assert "Error downloading voice" in result.output


class TestResolveVoiceAlias:
    """Tests for voice alias resolution."""

    def test_male_alias(self) -> None:
        """Resolve 'male' alias."""
        assert resolve_voice_alias("male") == "en_US-hfc_male-medium"

    def test_female_alias(self) -> None:
        """Resolve 'female' alias."""
        assert resolve_voice_alias("female") == "en_US-hfc_female-medium"

    def test_unknown_voice_unchanged(self) -> None:
        """Unknown voice names are returned unchanged."""
        assert resolve_voice_alias("en_US-lessac-medium") == "en_US-lessac-medium"

    def test_empty_string_unchanged(self) -> None:
        """Empty string is returned unchanged."""
        assert resolve_voice_alias("") == ""
