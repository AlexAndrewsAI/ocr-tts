"""Tests for the speak-region CLI bridge."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ocr_tts.ocr_region import Region
from ocr_tts.speak_region import app


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Typer CLI test runner."""
    return CliRunner()


class TestSpeakRegionCLI:
    """Tests for the speak-region command."""

    def test_app_is_typer(self, runner: CliRunner) -> None:
        """The speak_region app is a Typer instance."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    @patch("ocr_tts.speak_region.send_speak_request")
    @patch("ocr_tts.speak_region.extract_text", return_value="Hello world")
    @patch("ocr_tts.speak_region.image_is_blank", return_value=False)
    @patch("ocr_tts.speak_region.capture_selected_region")
    @patch("ocr_tts.speak_region.select_region")
    def test_speak_region_happy_path(
        self,
        mock_select: MagicMock,
        mock_capture: MagicMock,
        _mock_blank: MagicMock,
        _mock_extract: MagicMock,
        mock_speak: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Full happy path: region selected, text extracted, sent to speak."""
        mock_select.return_value = Region(x=0, y=0, width=100, height=50)
        mock_capture.return_value = MagicMock()
        mock_speak.return_value = {"status": "queued", "queue_size": 1}

        result = runner.invoke(app, [])

        assert result.exit_code == 0
        mock_select.assert_called_once()
        mock_capture.assert_called_once()
        _mock_extract.assert_called_once()
        mock_speak.assert_called_once_with(
            "Hello world",
            host="127.0.0.1",
            port=8000,
            voice="en_US-hfc_male-medium",
            speed=1.0,
            verbose=False,
        )
        assert "Hello world" in result.output

    @patch("ocr_tts.speak_region.send_speak_request")
    @patch("ocr_tts.speak_region.extract_text", return_value="Hello world")
    @patch("ocr_tts.speak_region.image_is_blank", return_value=False)
    @patch("ocr_tts.speak_region.capture_selected_region")
    @patch("ocr_tts.speak_region.select_region")
    def test_speak_region_custom_voice_speed_host(
        self,
        mock_select: MagicMock,
        mock_capture: MagicMock,
        _mock_blank: MagicMock,
        _mock_extract: MagicMock,
        mock_speak: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Voice, speed, host, and port are forwarded to the server."""
        mock_select.return_value = Region(x=0, y=0, width=100, height=50)
        mock_capture.return_value = MagicMock()
        mock_speak.return_value = {"status": "queued", "queue_size": 1}

        result = runner.invoke(
            app,
            [
                "-v",
                "fr_FR-siwis-medium",
                "-s",
                "1.5",
                "--host",
                "localhost",
                "--port",
                "9000",
            ],
        )

        assert result.exit_code == 0
        mock_speak.assert_called_once_with(
            "Hello world",
            host="localhost",
            port=9000,
            voice="fr_FR-siwis-medium",
            speed=1.5,
            verbose=False,
        )

    @patch("ocr_tts.speak_region.select_region")
    def test_speak_region_zero_size_region(
        self, mock_select: MagicMock, runner: CliRunner
    ) -> None:
        """A zero-size selection exits cleanly without calling speak."""
        mock_select.return_value = Region(x=0, y=0, width=0, height=0)

        result = runner.invoke(app, [])

        assert result.exit_code == 0
        assert "No region selected" in result.output

    @patch("ocr_tts.speak_region.send_speak_request")
    @patch("ocr_tts.speak_region.extract_text", return_value="")
    @patch("ocr_tts.speak_region.image_is_blank", return_value=False)
    @patch("ocr_tts.speak_region.capture_selected_region")
    @patch("ocr_tts.speak_region.select_region")
    def test_speak_region_no_text_detected(
        self,
        mock_select: MagicMock,
        mock_capture: MagicMock,
        _mock_blank: MagicMock,
        _mock_extract: MagicMock,
        mock_speak: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Empty OCR result exits with an error and does not call speak."""
        mock_select.return_value = Region(x=0, y=0, width=100, height=50)
        mock_capture.return_value = MagicMock()

        result = runner.invoke(app, [])

        assert result.exit_code == 1
        assert "no text detected" in result.output
        mock_speak.assert_not_called()

    @patch("ocr_tts.speak_region.send_speak_request")
    @patch("ocr_tts.speak_region.extract_text", return_value="Hello world")
    @patch("ocr_tts.speak_region.image_is_blank", return_value=False)
    @patch("ocr_tts.speak_region.capture_selected_region")
    @patch("ocr_tts.speak_region.select_region")
    def test_speak_region_save_image(
        self,
        mock_select: MagicMock,
        mock_capture: MagicMock,
        _mock_blank: MagicMock,
        _mock_extract: MagicMock,
        mock_speak: MagicMock,
        runner: CliRunner,
    ) -> None:
        """--save-image saves the captured region before OCR."""
        mock_image = MagicMock()
        mock_select.return_value = Region(x=0, y=0, width=100, height=50)
        mock_capture.return_value = mock_image
        mock_speak.return_value = {"status": "queued", "queue_size": 1}

        result = runner.invoke(app, ["--save-image", "region.png"])

        assert result.exit_code == 0
        mock_image.save.assert_called_once_with("region.png")
        assert "Saved captured region" in result.output

    @patch("ocr_tts.speak_region.send_speak_request")
    @patch("ocr_tts.speak_region.extract_text", return_value="Hello world")
    @patch("ocr_tts.speak_region.image_is_blank", return_value=True)
    @patch("ocr_tts.speak_region.capture_selected_region")
    @patch("ocr_tts.speak_region.select_region")
    def test_speak_region_blank_image_warning(
        self,
        mock_select: MagicMock,
        mock_capture: MagicMock,
        _mock_blank: MagicMock,
        _mock_extract: MagicMock,
        mock_speak: MagicMock,
        runner: CliRunner,
    ) -> None:
        """A blank image still proceeds to OCR and speak with a warning."""
        mock_select.return_value = Region(x=0, y=0, width=100, height=50)
        mock_capture.return_value = MagicMock()
        mock_speak.return_value = {"status": "queued", "queue_size": 1}

        result = runner.invoke(app, [])

        assert result.exit_code == 0
        assert "blank" in result.output.lower()
        mock_speak.assert_called_once()

    @patch("ocr_tts.speak_region.send_speak_request")
    @patch("ocr_tts.speak_region.extract_text", return_value="Bonjour")
    @patch("ocr_tts.speak_region.image_is_blank", return_value=False)
    @patch("ocr_tts.speak_region.capture_selected_region")
    @patch("ocr_tts.speak_region.select_region")
    def test_speak_region_uses_default_voice(
        self,
        mock_select: MagicMock,
        mock_capture: MagicMock,
        _mock_blank: MagicMock,
        _mock_extract: MagicMock,
        mock_speak: MagicMock,
        runner: CliRunner,
    ) -> None:
        """The default voice is used when --voice is not supplied."""
        mock_select.return_value = Region(x=0, y=0, width=100, height=50)
        mock_capture.return_value = MagicMock()
        mock_speak.return_value = {"status": "queued", "queue_size": 1}

        runner.invoke(app, [])

        mock_speak.assert_called_once_with(
            "Bonjour",
            host="127.0.0.1",
            port=8000,
            voice="en_US-hfc_male-medium",
            speed=1.0,
            verbose=False,
        )

    @patch("ocr_tts.speak_region.send_speak_request")
    @patch("ocr_tts.speak_region.extract_text", return_value="Hello world")
    @patch("ocr_tts.speak_region.image_is_blank", return_value=False)
    @patch("ocr_tts.speak_region.capture_selected_region")
    @patch("ocr_tts.speak_region.select_region")
    def test_speak_region_verbose_reports_latency(
        self,
        mock_select: MagicMock,
        mock_capture: MagicMock,
        _mock_blank: MagicMock,
        _mock_extract: MagicMock,
        mock_speak: MagicMock,
        runner: CliRunner,
    ) -> None:
        """--verbose prints a per-stage breakdown and adjusted turnaround.

        The user's region-select time is shown in the breakdown but is
        subtracted from the turnaround-time (which measures processing
        only, not interactive click-drag time).
        """
        mock_capture.return_value = MagicMock()
        mock_speak.return_value = {
            "status": "queued",
            "queue_size": 1,
            "synthesis_ms": 12.5,
            "latency_ms": 34.2,
        }

        def fake_select(
            on_handoff: Callable[[], None] | None = None,
        ) -> Region:
            # Simulate the real UI handing control to the user.
            if on_handoff is not None:
                on_handoff()
            return Region(x=0, y=0, width=100, height=50)

        mock_select.side_effect = fake_select

        result = runner.invoke(app, ["--verbose"])

        assert result.exit_code == 0
        mock_select.assert_called_once()
        assert mock_select.call_args.kwargs["on_handoff"] is not None
        mock_speak.assert_called_once_with(
            "Hello world",
            host="127.0.0.1",
            port=8000,
            voice="en_US-hfc_male-medium",
            speed=1.0,
            verbose=True,
        )
        # Per-stage breakdown is printed.
        assert "region-ui-load:" in result.output
        assert "user-region-select:" in result.output
        assert "capture:" in result.output
        assert "ocr:" in result.output
        assert "Latency: synthesis=12.5 ms" in result.output
        assert "piper-to-speech=34.2 ms" in result.output
        assert "turnaround-time:" in result.output

    @patch("ocr_tts.speak_region.send_speak_request")
    @patch("ocr_tts.speak_region.extract_text", return_value="Test")
    @patch("ocr_tts.speak_region.image_is_blank", return_value=False)
    @patch("ocr_tts.speak_region.capture_selected_region")
    @patch("ocr_tts.speak_region.select_region")
    def test_speak_region_uses_default_lang(
        self,
        mock_select: MagicMock,
        mock_capture: MagicMock,
        _mock_blank: MagicMock,
        mock_extract: MagicMock,
        _mock_speak: MagicMock,
        runner: CliRunner,
    ) -> None:
        """The default OCR language is used when --lang is not supplied."""
        mock_select.return_value = Region(x=0, y=0, width=100, height=50)
        mock_capture.return_value = MagicMock()

        runner.invoke(app, [])

        passed_config = mock_extract.call_args.kwargs["config"]
        assert passed_config.lang == "eng"

    @patch("ocr_tts.speak_region.send_speak_request")
    @patch("ocr_tts.speak_region.extract_text", return_value="Test")
    @patch("ocr_tts.speak_region.image_is_blank", return_value=False)
    @patch("ocr_tts.speak_region.capture_selected_region")
    @patch("ocr_tts.speak_region.select_region")
    def test_speak_region_custom_lang(
        self,
        mock_select: MagicMock,
        mock_capture: MagicMock,
        _mock_blank: MagicMock,
        mock_extract: MagicMock,
        _mock_speak: MagicMock,
        runner: CliRunner,
    ) -> None:
        """--lang is forwarded to the OCR config."""
        mock_select.return_value = Region(x=0, y=0, width=100, height=50)
        mock_capture.return_value = MagicMock()

        runner.invoke(app, ["--lang", "fra"])

        passed_config = mock_extract.call_args.kwargs["config"]
        assert passed_config.lang == "fra"
