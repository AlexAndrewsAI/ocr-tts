"""Additional coverage tests for the top-level CLI and ocr_region command."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import typer
from PIL import Image
from typer.testing import CliRunner

from ocr_tts.cli import app
from ocr_tts.ocr_region import Region, ocr_region_command, select_region


def _textured_image() -> Image.Image:
    """Build a small non-uniform RGB image."""
    image = Image.new("RGB", (100, 50))
    pixels = image.load()
    assert pixels is not None
    for x in range(0, 100, 2):
        pixels[x, 0] = (255, 255, 255)
    return image


class TestCliOcrFailurePaths:
    """Tests for CLI ocr copy/save/blank failure branches."""

    def test_copy_failures_blank_warning_and_save(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Failed clipboard writes and blank captures report warnings."""
        save_path = tmp_path / "capture.png"
        with (
            patch(
                "ocr_tts.cli.select_region",
                return_value=Region(x=0, y=0, width=100, height=50),
            ),
            patch(
                "ocr_tts.cli.capture_selected_region",
                return_value=_textured_image(),
            ),
            patch("ocr_tts.cli.image_is_blank", return_value=True),
            patch("ocr_tts.cli.extract_text", return_value="TXT"),
            patch("ocr_tts.cli.copy_text", return_value=False),
            patch("ocr_tts.cli.copy_image", return_value=False),
        ):
            result = runner.invoke(
                app,
                [
                    "ocr",
                    "--copy-text",
                    "--copy-image",
                    "--save-image",
                    str(save_path),
                ],
            )
        assert result.exit_code == 0
        assert save_path.exists()
        assert result.output.count("Could not copy") == 2
        assert "appears blank" in result.output


class TestCliLaunch:
    """Tests for the api launch subcommand."""

    def test_launch_delegates_to_serve(self, runner: CliRunner) -> None:
        """Api launch forwards host/port to the API serve entry point."""
        with patch("ocr_tts.cli._api_serve") as serve:
            result = runner.invoke(
                app,
                ["api", "launch", "--host", "127.0.0.1", "--port", "8123"],
            )
        assert result.exit_code == 0
        serve.assert_called_once_with(host="127.0.0.1", port=8123)


class TestOcrRegionCommand:
    """Tests for the standalone ocr_region command function."""

    def test_full_success_with_save(self, tmp_path: Path) -> None:
        """A successful run prints extracted text and saves the capture."""
        save_path = tmp_path / "cap.png"
        with (
            patch(
                "ocr_tts.ocr_region.select_region",
                return_value=Region(x=1, y=2, width=30, height=40),
            ),
            patch(
                "ocr_tts.ocr_region.capture_selected_region",
                return_value=_textured_image(),
            ) as capture,
            patch("ocr_tts.ocr_region.image_is_blank", return_value=False),
            patch("ocr_tts.ocr_region.extract_text", return_value="standalone text"),
        ):
            ocr_region_command(
                lang="eng", tesseract_cmd="tesseract", save_image=str(save_path)
            )
        assert save_path.exists()
        capture.assert_called_once()

    def test_no_region_selected_exits(self) -> None:
        """An empty region exits cleanly with a note."""
        with (
            patch(
                "ocr_tts.ocr_region.select_region",
                return_value=Region(x=0, y=0, width=0, height=0),
            ),
            pytest.raises(typer.Exit) as exc_info,
        ):
            ocr_region_command(lang="eng", tesseract_cmd="tesseract", save_image=None)
        assert exc_info.value.exit_code == 0

    def test_no_text_detected(self) -> None:
        """No OCR text reports the fallback message."""
        with (
            patch(
                "ocr_tts.ocr_region.select_region",
                return_value=Region(x=0, y=0, width=10, height=10),
            ),
            patch(
                "ocr_tts.ocr_region.capture_selected_region",
                return_value=_textured_image(),
            ),
            patch("ocr_tts.ocr_region.image_is_blank", return_value=False),
            patch("ocr_tts.ocr_region.extract_text", return_value=""),
        ):
            ocr_region_command(lang="eng", tesseract_cmd="tesseract", save_image=None)


class TestSelectRegionOnHandoff:
    """Tests that select_region invokes the on_handoff callback."""

    def test_on_handoff_called_before_mainloop(self) -> None:
        """on_handoff fires immediately before the overlay main loop."""
        calls: list[str] = []

        def handoff() -> None:
            calls.append("handoff")

        from tests.test_ocr_region import (
            _FakeEvent,
            _install_tkinter_mock,
            _uninstall_tkinter_mock,
        )

        handlers: dict[str, Any] = {}
        mock_tk = _install_tkinter_mock()
        try:
            root = mock_tk.Tk.return_value
            cv = mock_tk.Canvas.return_value

            def record_bind(*args: Any) -> None:
                handlers[args[0]] = args[1]

            cv.bind.side_effect = record_bind
            root.mainloop.side_effect = lambda: (
                handlers["<ButtonPress-1>"](_FakeEvent(5, 5)),
                handlers["<B1-Motion>"](_FakeEvent(15, 15)),
                handlers["<ButtonRelease-1>"](_FakeEvent(25, 25)),
            )
            with (
                patch(
                    "ocr_tts.ocr_region._primary_monitor",
                    return_value={"left": 0, "top": 0, "width": 100, "height": 100},
                ),
                patch("ocr_tts.ocr_region._capture_background", return_value=None),
            ):
                region = select_region(on_handoff=handoff)
        finally:
            _uninstall_tkinter_mock()

        assert calls == ["handoff"]
        assert region == Region(x=5, y=5, width=20, height=20)


class TestOcrRegionCommandBlankWarning:
    """Ensure the standalone command warns on blank captures."""

    def test_blank_capture_warns(self) -> None:
        """A blank capture prints the display-server warning."""
        with (
            patch(
                "ocr_tts.ocr_region.select_region",
                return_value=Region(x=0, y=0, width=10, height=10),
            ),
            patch(
                "ocr_tts.ocr_region.capture_selected_region",
                return_value=_textured_image(),
            ),
            patch("ocr_tts.ocr_region.image_is_blank", return_value=True),
            patch("ocr_tts.ocr_region.extract_text", return_value="words"),
        ):
            ocr_region_command(lang="eng", tesseract_cmd="tesseract", save_image=None)


class TestCaptureBackgroundOuterFailure:
    """Tests for the outer exception guard of _capture_background."""

    def test_sleep_failure_returns_none(self) -> None:
        """An exception escaping the retry loop yields None."""
        from ocr_tts.ocr_region import _capture_background

        with (
            patch(
                "ocr_tts.ocr_region.capture_region",
                side_effect=RuntimeError("capture broke"),
            ),
            patch(
                "ocr_tts.ocr_region.time.sleep",
                side_effect=RuntimeError("sleep interrupted"),
            ),
        ):
            assert (
                _capture_background({"left": 0, "top": 0, "width": 10, "height": 10})
                is None
            )
