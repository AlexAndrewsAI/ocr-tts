"""Tests for the OCR region module."""

import io
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dbus_fast import Variant
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from ocr_tts import desktop
from ocr_tts.cli import app
from ocr_tts.desktop import (
    CaptureError,
    _capture_region_wayland,
    _is_wayland,
    _monitor_bounds_tkinter,
    _portal_screenshot,
    capture_region,
    image_is_blank,
)
from ocr_tts.ocr_region import (
    OCRConfig,
    Region,
    _capture_background,
    _crop_selection_background,
    _primary_monitor,
    capture_selected_region,
    extract_text,
    select_region,
)


@pytest.fixture(autouse=True)
def _reset_portal_cache() -> Any:
    """Reset the portal-unavailable cache around every test."""
    import ocr_tts.desktop as mod

    mod._portal_unavailable = False
    yield
    mod._portal_unavailable = False


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Typer CLI test runner."""
    return CliRunner()


class TestRegion:
    """Tests for the Region namedtuple."""

    def test_region_creation(self) -> None:
        """Test creating a Region instance."""
        region = Region(x=10, y=20, width=100, height=50)
        assert region.x == 10
        assert region.y == 20
        assert region.width == 100
        assert region.height == 50


class TestOCRConfig:
    """Tests for OCRConfig model."""

    def test_default_values(self) -> None:
        """Test default OCR config values."""
        config = OCRConfig()
        assert config.tesseract_cmd == "tesseract"
        assert config.lang == "eng"

    def test_custom_values(self) -> None:
        """Test custom OCR config values."""
        config = OCRConfig(tesseract_cmd="/usr/bin/tesseract", lang="fra")
        assert config.tesseract_cmd == "/usr/bin/tesseract"
        assert config.lang == "fra"

    def test_frozen_immutability(self) -> None:
        """Test that OCRConfig is frozen and cannot be modified."""
        config = OCRConfig()
        with pytest.raises(ValidationError, match="Instance is frozen"):
            config.lang = "deu"  # type: ignore[misc]


_MONITOR = {"left": 0, "top": 0, "width": 1920, "height": 1080}


class _FakeTclError(Exception):
    """Stand-in for ``tkinter.TclError`` in tests."""


class _FakeEvent:
    """Minimal canvas event with x/y coordinates."""

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


def _install_tkinter_mock() -> MagicMock:
    """Install a mock ``tkinter`` module in sys.modules."""
    import sys

    mock_tk = MagicMock()
    mock_tk.TclError = _FakeTclError
    sys.modules["tkinter"] = mock_tk
    return mock_tk


def _uninstall_tkinter_mock() -> None:
    """Remove the mock ``tkinter`` module from sys.modules."""
    import sys

    sys.modules.pop("tkinter", None)


def _canvas_handlers(canvas: MagicMock) -> dict[str, Any]:
    """Extract event handlers bound to a mock canvas."""
    handlers: dict[str, Any] = {}
    for args in canvas.bind.call_args_list:
        handlers[args[0][0]] = args[0][1]
    return handlers


class TestSelectRegion:
    """Tests for select_region function."""

    def _simulate_drag(self, canvas: MagicMock) -> None:
        """Simulate a complete click-drag-release cycle."""
        handlers = _canvas_handlers(canvas)
        handlers["<ButtonPress-1>"](_FakeEvent(10, 20))
        handlers["<B1-Motion>"](_FakeEvent(60, 70))
        handlers["<ButtonRelease-1>"](_FakeEvent(110, 120))

    def test_select_region_returns_region(self) -> None:
        """Test that select_region returns a valid Region."""
        mock_tk = _install_tkinter_mock()
        try:
            mock_root = mock_tk.Tk.return_value
            mock_canvas = mock_tk.Canvas.return_value
            with (
                patch("ocr_tts.ocr_region._primary_monitor", return_value=_MONITOR),
                patch("ocr_tts.ocr_region._capture_background", return_value=None),
            ):
                mock_root.mainloop.side_effect = lambda: self._simulate_drag(
                    mock_canvas
                )
                region = select_region()
        finally:
            _uninstall_tkinter_mock()

        assert isinstance(region, Region)
        assert region.x == 10
        assert region.y == 20
        assert region.width == 100
        assert region.height == 100
        mock_root.destroy.assert_called_once()

    def test_select_region_offsets_monitor_coordinates(self) -> None:
        """Test that canvas coordinates are offset by the monitor origin."""
        mock_tk = _install_tkinter_mock()
        try:
            mock_root = mock_tk.Tk.return_value
            mock_canvas = mock_tk.Canvas.return_value
            monitor = {"left": 100, "top": 50, "width": 1920, "height": 1080}
            with (
                patch("ocr_tts.ocr_region._primary_monitor", return_value=monitor),
                patch("ocr_tts.ocr_region._capture_background", return_value=None),
            ):
                mock_root.mainloop.side_effect = lambda: self._simulate_drag(
                    mock_canvas
                )
                region = select_region()
        finally:
            _uninstall_tkinter_mock()

        assert region == Region(x=110, y=70, width=100, height=100)

    def test_select_region_escape_cancels(self) -> None:
        """Test that Escape cancels the selection with an empty region."""
        mock_tk = _install_tkinter_mock()
        try:
            mock_root = mock_tk.Tk.return_value
            mock_canvas = mock_tk.Canvas.return_value
            with (
                patch("ocr_tts.ocr_region._primary_monitor", return_value=_MONITOR),
                patch("ocr_tts.ocr_region._capture_background", return_value=None),
            ):

                def simulate_escape() -> None:
                    handlers = _canvas_handlers(mock_canvas)
                    handlers["<ButtonPress-1>"](_FakeEvent(10, 20))
                    handlers["<Escape>"](_FakeEvent(0, 0))

                mock_root.mainloop.side_effect = simulate_escape
                region = select_region()
        finally:
            _uninstall_tkinter_mock()

        assert region == Region(x=0, y=0, width=0, height=0)

    def test_select_region_release_without_press(self) -> None:
        """Test that a release with no preceding press yields an empty region."""
        mock_tk = _install_tkinter_mock()
        try:
            mock_root = mock_tk.Tk.return_value
            mock_canvas = mock_tk.Canvas.return_value
            with (
                patch("ocr_tts.ocr_region._primary_monitor", return_value=_MONITOR),
                patch("ocr_tts.ocr_region._capture_background", return_value=None),
            ):

                def simulate_release() -> None:
                    handlers = _canvas_handlers(mock_canvas)
                    handlers["<ButtonRelease-1>"](_FakeEvent(10, 20))

                mock_root.mainloop.side_effect = simulate_release
                region = select_region()
        finally:
            _uninstall_tkinter_mock()

        assert region == Region(x=0, y=0, width=0, height=0)

    def test_select_region_with_background(self) -> None:
        """Test that a captured background is drawn as the overlay image."""
        import sys

        mock_tk = _install_tkinter_mock()
        mock_imagetk = MagicMock()
        sys.modules["PIL.ImageTk"] = mock_imagetk
        try:
            mock_root = mock_tk.Tk.return_value
            mock_canvas = mock_tk.Canvas.return_value
            background = Image.new("RGB", (10, 10))
            with (
                patch("ocr_tts.ocr_region._primary_monitor", return_value=_MONITOR),
                patch(
                    "ocr_tts.ocr_region._capture_background", return_value=background
                ),
            ):
                mock_root.mainloop.side_effect = lambda: self._simulate_drag(
                    mock_canvas
                )
                region = select_region()
        finally:
            sys.modules.pop("PIL.ImageTk", None)
            _uninstall_tkinter_mock()

        assert region == Region(x=10, y=20, width=100, height=100)
        mock_imagetk.PhotoImage.assert_called_once()
        displayed = mock_imagetk.PhotoImage.call_args[0][0]
        assert displayed.size == (1920, 1080)
        mock_canvas.create_image.assert_called_once()

    def test_select_region_no_tkinter(self) -> None:
        """Test that a missing tkinter module yields an empty region."""
        import sys

        with patch.dict(sys.modules, {"tkinter": None}):
            region = select_region()

        assert region == Region(x=0, y=0, width=0, height=0)

    def test_select_region_no_display(self) -> None:
        """Test that an unavailable display yields an empty region."""
        mock_tk = _install_tkinter_mock()
        try:
            mock_tk.Tk.side_effect = _FakeTclError("no display available")
            with (
                patch("ocr_tts.ocr_region._primary_monitor", return_value=_MONITOR),
                patch("ocr_tts.ocr_region._capture_background", return_value=None),
            ):
                region = select_region()
        finally:
            _uninstall_tkinter_mock()

        assert region == Region(x=0, y=0, width=0, height=0)

    def test_select_region_no_monitor(self) -> None:
        """Test that a monitor lookup failure yields an empty region."""
        _install_tkinter_mock()
        try:
            with patch(
                "ocr_tts.ocr_region._primary_monitor",
                side_effect=OSError("no screen"),
            ):
                region = select_region()
        finally:
            _uninstall_tkinter_mock()

        assert region == Region(x=0, y=0, width=0, height=0)

    def test_capture_background_returns_image(self) -> None:
        """Test that capture_background returns the captured image."""
        image = Image.new("RGB", (10, 10), color=(255, 255, 255))
        pixels = image.load()
        assert pixels is not None
        pixels[0, 0] = (0, 0, 0)
        with patch("ocr_tts.ocr_region.capture_region", return_value=image):
            result = _capture_background(_MONITOR)

        assert result == image

    def test_capture_background_returns_none_on_error(self) -> None:
        """Test that capture_background returns None when capture fails."""
        with patch(
            "ocr_tts.ocr_region.capture_region",
            side_effect=OSError("capture failed"),
        ):
            result = _capture_background(_MONITOR)

        assert result is None

    def test_capture_background_returns_none_when_blank(self) -> None:
        """Test that capture_background returns None for a blank frame."""
        image = Image.new("RGB", (10, 10), color=(0, 0, 0))
        with patch("ocr_tts.ocr_region.capture_region", return_value=image):
            result = _capture_background(_MONITOR)

        assert result is None

    def test_image_is_blank_solid_color(self) -> None:
        """Test that a uniform image is detected as blank."""
        image = Image.new("RGB", (50, 50), color=(0, 0, 0))
        assert image_is_blank(image) is True

    def test_image_is_blank_near_uniform(self) -> None:
        """Test that a low-contrast image is detected as blank."""
        image = Image.new("L", (20, 20), color=128)
        assert image_is_blank(image) is True

    def test_image_is_blank_not_blank(self) -> None:
        """Test that a textured image is not detected as blank."""
        image = Image.new("L", (20, 20), color=0)
        pixels = image.load()
        assert pixels is not None
        for x in range(image.width):
            pixels[x, x % image.height] = 255
        assert image_is_blank(image) is False

    def test_primary_monitor_returns_bounds(self) -> None:
        """Test that primary_monitor returns the primary monitor bounds."""
        mock_sct = MagicMock()
        mock_sct.monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
        ]
        mock_sct.__enter__.return_value = mock_sct
        with patch("ocr_tts.desktop.mss.mss", return_value=mock_sct):
            monitor = _primary_monitor()

        assert monitor == _MONITOR


class TestCaptureRegion:
    """Tests for capture_region function."""

    def _mock_sct(self) -> MagicMock:
        """Build a mocked mss context manager returning a varied screenshot."""
        base = Image.new("RGB", (200, 100))
        pixels = base.load()
        assert pixels is not None
        for x in range(200):
            for y in range(100):
                pixels[x, y] = ((x * 7) % 256, (y * 5) % 256, 128)

        mock_screenshot = MagicMock()
        mock_screenshot.size = (200, 100)
        mock_screenshot.rgb = base.tobytes()

        mock_sct = MagicMock()
        mock_sct.__enter__ = MagicMock(return_value=mock_sct)
        mock_sct.__exit__ = MagicMock(return_value=False)
        mock_sct.grab.return_value = mock_screenshot
        return mock_sct

    def test_capture_region_uses_x11_backend(self) -> None:
        """Test that capture_region returns an RGB image via the X11 backend."""
        mock_sct = self._mock_sct()
        with (
            patch("ocr_tts.desktop._is_wayland", return_value=False),
            patch("ocr_tts.desktop.mss.mss", return_value=mock_sct),
        ):
            image = capture_region(0, 0, 200, 100)

        assert image.size == (200, 100)
        assert image.mode == "RGB"
        mock_sct.grab.assert_called_once_with(
            {"left": 0, "top": 0, "width": 200, "height": 100}
        )

    def test_capture_region_uses_wayland_backend(self) -> None:
        """Test that capture_region returns the portal backend result."""
        mock_image = Image.new("RGB", (200, 100), color=(1, 2, 3))
        with (
            patch("ocr_tts.desktop._is_wayland", return_value=True),
            patch("ocr_tts.desktop._portal_unavailable", False),
            # Patch desktop's blank check (not ocr_region's copy of the
            # name) so the solid-color mock is treated as a valid frame.
            patch("ocr_tts.desktop.image_is_blank", return_value=False),
            # Prevent real screenshot tools on machines that have them.
            patch("ocr_tts.desktop.shutil.which", return_value=None),
            patch(
                "ocr_tts.desktop._capture_region_wayland",
                return_value=mock_image,
            ) as mock_wayland,
        ):
            image = capture_region(0, 0, 200, 100)

        assert image.size == mock_image.size
        assert list(image.getdata()) == list(mock_image.getdata())
        mock_wayland.assert_called_once_with(0, 0, 200, 100)

    def test_capture_region_falls_back_from_wayland(self) -> None:
        """Test that capture_region falls back to X11 when the portal fails."""
        mock_sct = self._mock_sct()
        with (
            patch("ocr_tts.desktop._is_wayland", return_value=True),
            patch("ocr_tts.desktop.time.sleep"),
            patch("ocr_tts.desktop.shutil.which", return_value=None),
            patch(
                "ocr_tts.desktop._capture_region_wayland",
                side_effect=RuntimeError("no portal"),
            ),
            patch("ocr_tts.desktop.mss.mss", return_value=mock_sct),
        ):
            image = capture_region(0, 0, 200, 100)

        assert image.size == (200, 100)

    def test_capture_region_wayland_crops_full_screen(self) -> None:
        """Test that the Wayland backend crops the region from the full screen."""
        full = Image.new("RGB", (1920, 1080), color=(10, 20, 30))
        with (
            patch("ocr_tts.desktop._portal_screenshot", return_value=full),
            patch(
                "ocr_tts.desktop._primary_monitor",
                return_value=_MONITOR,
            ),
        ):
            image = _capture_region_wayland(10, 20, 100, 50)

        assert image.size == (100, 50)
        assert image.getpixel((0, 0)) == (10, 20, 30)

    def test_capture_region_wayland_offsets_by_monitor_origin(self) -> None:
        """Test that the Wayland backend offsets by the monitor origin."""
        full = Image.new("RGB", (1920, 1080), color=(0, 0, 0))
        full.putpixel((50, 110), (255, 0, 0))
        monitor = {"left": 100, "top": 50, "width": 1920, "height": 1080}
        with (
            patch("ocr_tts.desktop._portal_screenshot", return_value=full),
            patch("ocr_tts.desktop._primary_monitor", return_value=monitor),
        ):
            image = _capture_region_wayland(110, 70, 100, 100)

        assert image.size == (100, 100)
        assert image.getpixel((40, 90)) == (255, 0, 0)

    def test_capture_fullscreen_scales_to_frame_resolution(self) -> None:
        """Test cropping scales logical coords to a higher-res frame."""
        full = Image.new("RGB", (3840, 2160), color=(0, 0, 0))
        # red square at scaled position of logical (200..300, 200..300)
        for x in range(400, 600):
            for y in range(400, 600):
                full.putpixel((x, y), (255, 0, 0))
        monitor = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        with (
            patch(
                "ocr_tts.desktop._primary_monitor",
                return_value=monitor,
            ),
        ):
            image = desktop._crop_fullscreen_to_region(full, 250, 250, 100, 50)

        assert image.size == (200, 100)
        # logical (250,250) maps to frame (500,500), inside the red square
        assert image.getpixel((5, 5)) == (255, 0, 0)
        # logical (330,270) maps to (660,540), outside the red square
        assert image.getpixel((160, 40)) == (0, 0, 0)


class TestCaptureSelectedRegion:
    """Tests for capture_selected_region frozen-frame policy."""

    def _textured(self, size: tuple[int, int], pixel: tuple[int, int]) -> Image.Image:
        """Build a gradient RGB image with a distinguishing pixel."""
        image = Image.new("RGB", size)
        pixels = image.load()
        assert pixels is not None
        for x in range(size[0]):
            for y in range(size[1]):
                pixels[x, y] = ((x * 3) % 256, (y * 3) % 256, 30)
        image.putpixel(pixel, (255, 0, 0))
        return image

    def test_capture_selected_region_prefers_overlay_frame(self) -> None:
        """The frozen overlay frame wins even when a fresh capture works."""
        background = self._textured((1920, 1080), (50, 70))
        monitor = {"left": 100, "top": 50, "width": 1920, "height": 1080}
        with (
            patch("ocr_tts.ocr_region.capture_region") as mock_capture,
            patch("ocr_tts.ocr_region._selection_background", background),
            patch("ocr_tts.ocr_region._selection_monitor", monitor),
        ):
            image = capture_selected_region(Region(110, 70, 100, 100))

        mock_capture.assert_not_called()
        assert image.size == (100, 100)
        assert image.getpixel((40, 50)) == (255, 0, 0)

    def test_capture_selected_region_uses_fresh_capture(self) -> None:
        """A fresh capture happens when no overlay frame exists."""
        fresh = self._textured((100, 50), (5, 5))
        with (
            patch(
                "ocr_tts.ocr_region.capture_region", return_value=fresh
            ) as mock_capture,
            patch("ocr_tts.ocr_region._selection_background", None),
        ):
            image = capture_selected_region(Region(0, 0, 100, 50))

        mock_capture.assert_called_once_with(0, 0, 100, 50)
        assert image == fresh

    def test_capture_selected_region_skips_blank_overlay_frame(self) -> None:
        """A blank frozen frame defers to a fresh capture."""
        fresh = self._textured((100, 50), (5, 5))
        blank_background = Image.new("RGB", (1920, 1080))
        monitor = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        with (
            patch("ocr_tts.ocr_region.capture_region", return_value=fresh),
            patch("ocr_tts.ocr_region._selection_background", blank_background),
            patch("ocr_tts.ocr_region._selection_monitor", monitor),
        ):
            image = capture_selected_region(Region(0, 0, 100, 50))

        assert image == fresh

    def test_capture_selected_region_skips_out_of_bounds_overlay_frame(self) -> None:
        """An unmappable frozen frame defers to a fresh capture."""
        fresh = self._textured((100, 50), (5, 5))
        background = Image.new("RGB", (1920, 1080))
        monitor = {"left": 100, "top": 50, "width": 1920, "height": 1080}
        with (
            patch("ocr_tts.ocr_region.capture_region", return_value=fresh),
            patch("ocr_tts.ocr_region._selection_background", background),
            patch("ocr_tts.ocr_region._selection_monitor", monitor),
        ):
            # Starts left of the monitor origin, so cropping cannot map.
            image = capture_selected_region(Region(50, 70, 100, 100))

        assert image == fresh

    def test_capture_selected_region_keeps_blank_when_no_overlay_frame(self) -> None:
        """Test that the blank capture is returned when no overlay frame exists."""
        blank = Image.new("RGB", (100, 50), color=(0, 0, 0))
        with (
            patch("ocr_tts.ocr_region.capture_region", return_value=blank),
            patch("ocr_tts.ocr_region._selection_background", None),
            patch("ocr_tts.ocr_region._selection_monitor", _MONITOR),
        ):
            image = capture_selected_region(Region(0, 0, 100, 50))

        assert image == blank

    def test_crop_selection_background_returns_none_without_background(self) -> None:
        """Test that cropping returns None when no background was captured."""
        with patch("ocr_tts.ocr_region._selection_background", None):
            assert _crop_selection_background(Region(0, 0, 10, 10)) is None

    def test_crop_selection_background_returns_none_outside_bounds(self) -> None:
        """Test that cropping returns None when the region is out of bounds."""
        background = Image.new("RGB", (1920, 1080))
        monitor = {"left": 100, "top": 50, "width": 1920, "height": 1080}
        with (
            patch("ocr_tts.ocr_region._selection_background", background),
            patch("ocr_tts.ocr_region._selection_monitor", monitor),
        ):
            region = Region(50, 70, 100, 100)
            assert _crop_selection_background(region) is None

    def test_crop_selection_background_scales_to_frame_resolution(self) -> None:
        """Test that cropping maps logical coords into a higher-res frame."""
        background = Image.new("RGB", (3840, 2160), color=(0, 0, 0))
        # red pixel at background position of logical (120, 120)
        background.putpixel((240, 240), (255, 0, 0))
        monitor = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        with (
            patch(
                "ocr_tts.ocr_region._selection_background",
                background,
            ),
            patch("ocr_tts.ocr_region._selection_monitor", monitor),
        ):
            image = _crop_selection_background(Region(100, 100, 100, 100))

        assert image is not None
        assert image.size == (200, 200)
        assert image.getpixel((40, 40)) == (255, 0, 0)


class TestPortalScreenshot:
    """Tests for _portal_screenshot function."""

    def _install_dbus_mocks(self) -> tuple[MagicMock, MagicMock]:
        """Install mocked dbus_fast modules in sys.modules."""
        import sys

        from dbus_fast import Variant

        mock_dbus_fast = MagicMock()
        mock_dbus_fast.BusType.SESSION = "SESSION"
        mock_dbus_fast.Message = MagicMock()
        mock_dbus_fast.MessageType.METHOD_RETURN = "METHOD_RETURN"
        mock_dbus_fast.MessageType.ERROR = "ERROR"
        mock_dbus_fast.Variant = Variant
        mock_aio = MagicMock()
        mock_dbus_fast.aio = mock_aio
        sys.modules["dbus_fast"] = mock_dbus_fast
        sys.modules["dbus_fast.aio"] = mock_aio
        return mock_dbus_fast, mock_aio

    def _uninstall_dbus_mocks(self) -> None:
        """Remove mocked dbus_fast modules from sys.modules."""
        import sys

        sys.modules.pop("dbus_fast", None)
        sys.modules.pop("dbus_fast.aio", None)

    @staticmethod
    def _await_call_result(value: Any) -> Any:
        """Return a side effect that resolves an awaited call to value."""

        async def fake_call(*_args: Any, **_kwargs: Any) -> Any:
            return value

        return fake_call

    def test_portal_screenshot_returns_image(self, tmp_path: Any) -> None:
        """Test that the portal screenshot loads the returned image."""
        _, mock_aio = self._install_dbus_mocks()
        try:
            mock_bus = MagicMock()
            mock_bus.connect = AsyncMock(return_value=mock_bus)
            mock_aio.MessageBus.return_value = mock_bus

            handle = "/org/freedesktop/portal/desktop/request/1_2/ocr_region_1"
            reply = MagicMock()
            reply.message_type = "METHOD_RETURN"
            reply.body = [handle]
            mock_bus.call.side_effect = self._await_call_result(reply)

            png = tmp_path / "shot.png"
            Image.new("RGB", (10, 10), color=(255, 0, 0)).save(png)

            def respond(handler: Any) -> None:
                """Deliver the portal Response signal synchronously."""
                response = MagicMock()
                response.interface = "org.freedesktop.portal.Request"
                response.member = "Response"
                response.path = handle
                response.body = [0, {"uri": Variant("s", f"file://{png}")}]
                handler(response)
                return None

            mock_bus.add_message_handler.side_effect = respond

            image = _portal_screenshot()
        finally:
            self._uninstall_dbus_mocks()

        assert image.size == (10, 10)
        assert image.getpixel((0, 0)) == (255, 0, 0)

    def test_portal_screenshot_rejects_error_reply(self) -> None:
        """Test that an error reply raises a RuntimeError."""
        _, mock_aio = self._install_dbus_mocks()
        try:
            mock_bus = MagicMock()
            mock_bus.connect = AsyncMock(return_value=mock_bus)
            mock_aio.MessageBus.return_value = mock_bus
            reply = MagicMock()
            reply.message_type = "ERROR"
            reply.error_name = "org.freedesktop.DBus.Error.UnknownMethod"
            mock_bus.call.side_effect = self._await_call_result(reply)

            with pytest.raises(RuntimeError, match="Portal Screenshot call failed"):
                _portal_screenshot()
        finally:
            self._uninstall_dbus_mocks()

    def test_portal_screenshot_rejects_nonzero_response(self) -> None:
        """Test that a non-zero response code raises a RuntimeError."""
        _, mock_aio = self._install_dbus_mocks()
        try:
            mock_bus = MagicMock()
            mock_bus.connect = AsyncMock(return_value=mock_bus)
            mock_aio.MessageBus.return_value = mock_bus

            handle = "/org/freedesktop/portal/desktop/request/1_2/ocr_region_1"
            reply = MagicMock()
            reply.message_type = "METHOD_RETURN"
            reply.body = [handle]
            mock_bus.call.side_effect = self._await_call_result(reply)

            def respond(handler: Any) -> None:
                """Deliver a cancelled Response signal."""
                response = MagicMock()
                response.interface = "org.freedesktop.portal.Request"
                response.member = "Response"
                response.path = handle
                response.body = [1, {}]
                handler(response)
                return None

            mock_bus.add_message_handler.side_effect = respond

            with pytest.raises(RuntimeError, match="response code 1"):
                _portal_screenshot()
        finally:
            self._uninstall_dbus_mocks()

    def test_portal_screenshot_without_dbus_fast(self) -> None:
        """Test that a missing dbus_fast module raises a RuntimeError."""
        import sys

        with (
            patch.dict(sys.modules, {"dbus_fast": None, "dbus_fast.aio": None}),
            pytest.raises(RuntimeError, match="dbus-fast is required"),
        ):
            _portal_screenshot()


class TestCaptureBackends:
    """Tests for the multi-backend capture chain."""

    def _png_bytes(self) -> bytes:
        """Encode a small non-blank PNG into bytes."""
        image = Image.new("RGB", (20, 20), color=(0, 0, 0))
        for x in range(10):
            for y in range(20):
                image.putpixel((x, y), (255, 255, 255))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_capture_region_raises_capture_error_when_all_fail(self) -> None:
        """Test that a CaptureError aggregates per-backend failures."""
        with (
            patch("ocr_tts.desktop._is_wayland", return_value=False),
            patch("ocr_tts.desktop.time.sleep"),
            patch("ocr_tts.desktop.shutil.which", return_value=None),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ocr_tts.desktop.mss.mss",
                side_effect=OSError("X11 Protocol Error"),
            ),
            pytest.raises(CaptureError) as exc_info,
        ):
            capture_region(0, 0, 100, 50)

        assert "mss/x11" in " ".join(exc_info.value.errors)

    def test_capture_region_uses_grim_when_portal_and_mss_fail(self) -> None:
        """Test that grim grabs full output and scales when portal/mss fail."""
        frame = Image.new("RGB", (30, 30), color=(0, 0, 0))
        for x in range(15):
            for y in range(30):
                frame.putpixel((x, y), (255, 255, 255))
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        png = buffer.getvalue()

        def fake_run(*_args: Any, **_kwargs: Any) -> MagicMock:
            proc = MagicMock()
            proc.returncode = 0
            proc.stdout = png
            proc.stderr = b""
            return proc

        monitor = {"left": 0, "top": 0, "width": 30, "height": 30}
        with (
            patch("ocr_tts.desktop._is_wayland", return_value=True),
            patch("ocr_tts.desktop.time.sleep"),
            patch("ocr_tts.desktop.shutil.which", return_value="/usr/bin/grim"),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ocr_tts.desktop._capture_region_wayland",
                side_effect=RuntimeError(
                    "Portal Screenshot call failed: "
                    "org.freedesktop.DBus.Error.UnknownMethod"
                ),
            ),
            patch("ocr_tts.desktop.mss.mss", side_effect=OSError("X11 Protocol Error")),
            patch("ocr_tts.desktop._primary_monitor", return_value=monitor),
            patch("ocr_tts.desktop.subprocess.run", side_effect=fake_run) as mock_run,
        ):
            image = capture_region(5, 6, 20, 20)

        assert image.size == (20, 20)
        # Logical (5,6) maps to frame (5,6): inside the white left half.
        assert image.getpixel((0, 0)) == (255, 255, 255)
        # Logical (24,6) maps to frame (24,6): in the black right half.
        assert image.getpixel((19, 0)) == (0, 0, 0)
        cmd = mock_run.call_args[0][0]
        assert cmd[:2] == ["/usr/bin/grim", "-"]

    def test_custom_command_backend_used_first(self) -> None:
        """Test that OCR_TTS_CAPTURE_COMMAND takes precedence."""

        def fake_run(*_args: Any, **_kwargs: Any) -> MagicMock:
            proc = MagicMock()
            proc.returncode = 0
            proc.stdout = self._png_bytes()
            proc.stderr = b""
            return proc

        with (
            patch("ocr_tts.desktop._is_wayland", return_value=False),
            patch.dict(
                os.environ,
                {"OCR_TTS_CAPTURE_COMMAND": "fakeshot --rect {x},{y},{w},{h}"},
            ),
            patch("ocr_tts.desktop.subprocess.run", side_effect=fake_run) as mock_run,
        ):
            image = capture_region(1, 2, 3, 4)

        assert image.size == (20, 20)
        cmd = mock_run.call_args[0][0]
        assert cmd == ["fakeshot", "--rect", "1,2,3,4"]

    def test_capture_region_all_blank_returns_first_blank(self) -> None:
        """Test that a blank-but-successful capture is still returned."""
        blank = Image.new("RGB", (30, 30), color=(7, 7, 7))
        buffer = io.BytesIO()
        blank.save(buffer, format="PNG")
        png = buffer.getvalue()

        def fake_run(*_args: Any, **_kwargs: Any) -> MagicMock:
            proc = MagicMock()
            proc.returncode = 0
            proc.stdout = png
            proc.stderr = b""
            return proc

        with (
            patch("ocr_tts.desktop._is_wayland", return_value=False),
            patch("ocr_tts.desktop.shutil.which", return_value=None),
            patch.dict(
                os.environ,
                {"OCR_TTS_CAPTURE_COMMAND": "blanksnap"},
            ),
            patch("ocr_tts.desktop.subprocess.run", side_effect=fake_run) as mock_run,
        ):
            image = capture_region(0, 0, 30, 30)

        # custom command produced a decodable blank PNG and is returned
        # even though it is blank (no backend produced a better frame).
        assert image.size == (30, 30)
        mock_run.assert_called_once()


class TestMonitorBoundsTkinter:
    """Tests for the tkinter-based monitor bounds fallback."""

    def test_monitor_bounds_via_tkinter(self) -> None:
        """Test that tkinter screen size is used when mss fails."""
        mock_tk = _install_tkinter_mock()
        try:
            mock_root = mock_tk.Tk.return_value
            mock_root.winfo_screenwidth.return_value = 1280
            mock_root.winfo_screenheight.return_value = 720
            bounds = _monitor_bounds_tkinter()
        finally:
            _uninstall_tkinter_mock()

        assert bounds == {"left": 0, "top": 0, "width": 1280, "height": 720}
        mock_root.destroy.assert_called_once()

    def test_primary_monitor_falls_back_to_tkinter(self) -> None:
        """Test that primary_monitor uses tkinter when mss fails."""
        mock_tk = _install_tkinter_mock()
        try:
            mock_root = mock_tk.Tk.return_value
            mock_root.winfo_screenwidth.return_value = 800
            mock_root.winfo_screenheight.return_value = 600
            with patch(
                "ocr_tts.desktop.mss.mss",
                side_effect=OSError("no X display"),
            ):
                monitor = _primary_monitor()
        finally:
            _uninstall_tkinter_mock()

        assert monitor == {"left": 0, "top": 0, "width": 800, "height": 600}

    def test_primary_monitor_tkinter_failure_propagates(self) -> None:
        """Test that a tkinter probe failure raises through."""
        mock_tk = _install_tkinter_mock()
        try:
            mock_tk.Tk.side_effect = _FakeTclError("no display")
            with (
                patch(
                    "ocr_tts.desktop.mss.mss",
                    side_effect=OSError("no X display"),
                ),
                pytest.raises(RuntimeError, match="tkinter screen probe"),
            ):
                _primary_monitor()
        finally:
            _uninstall_tkinter_mock()


class TestPortalUnavailableCache:
    """Tests for caching of portal UnknownMethod failures."""

    def test_unknown_method_disables_portal_for_later_calls(self) -> None:
        """Test that an UnknownMethod portal error is cached."""
        import ocr_tts.desktop as mod

        def fail_wayland(*_args: Any) -> Image.Image:
            raise RuntimeError(
                "Portal Screenshot call failed: "
                "org.freedesktop.DBus.Error.UnknownMethod"
            )

        with (
            patch("ocr_tts.desktop._is_wayland", return_value=True),
            patch("ocr_tts.desktop.time.sleep"),
            patch("ocr_tts.desktop.shutil.which", return_value=None),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ocr_tts.desktop._capture_region_wayland",
                side_effect=fail_wayland,
            ),
            patch("ocr_tts.desktop.mss.mss", side_effect=OSError("bad X")),
            pytest.raises(CaptureError),
        ):
            capture_region(0, 0, 10, 10)

        assert mod._portal_unavailable is True

        with (
            patch("ocr_tts.desktop._capture_region_wayland") as mock_wayland,
            patch("ocr_tts.desktop.shutil.which", return_value=None),
            patch("ocr_tts.desktop.mss.mss", side_effect=OSError("bad X")),
            pytest.raises(CaptureError),
        ):
            capture_region(0, 0, 10, 10)

        mock_wayland.assert_not_called()

    def test_other_portal_errors_are_not_cached(self) -> None:
        """Test that transient portal errors do not disable the backend."""
        import ocr_tts.desktop as mod

        with (
            patch("ocr_tts.desktop._is_wayland", return_value=True),
            patch("ocr_tts.desktop.time.sleep"),
            patch("ocr_tts.desktop.shutil.which", return_value=None),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ocr_tts.desktop._capture_region_wayland",
                side_effect=RuntimeError("portal busy"),
            ),
            patch("ocr_tts.desktop.mss.mss", side_effect=OSError("bad X")),
            pytest.raises(CaptureError),
        ):
            capture_region(0, 0, 10, 10)

        assert mod._portal_unavailable is False


class TestCaptureSelectedRegionBackendFailure:
    """Tests for capture_selected_region handling of total capture failure."""

    def test_total_failure_falls_back_to_overlay_frame(self) -> None:
        """Test that a CaptureError uses the overlay frame if usable."""
        background = Image.new("RGB", (1920, 1080))
        pixels = background.load()
        assert pixels is not None
        for x in range(0, 1920, 4):
            for y in range(0, 1080, 4):
                pixels[x, y] = ((x * 3) % 256, (y * 3) % 256, 30)
        background.putpixel((50, 110), (255, 0, 0))
        monitor = {"left": 100, "top": 50, "width": 1920, "height": 1080}
        with (
            patch(
                "ocr_tts.ocr_region.capture_region",
                side_effect=CaptureError(["mss/x11: boom"]),
            ),
            patch("ocr_tts.ocr_region._selection_background", background),
            patch("ocr_tts.ocr_region._selection_monitor", monitor),
        ):
            image = capture_selected_region(Region(110, 70, 100, 100))

        assert image.size == (100, 100)
        assert image.getpixel((40, 90)) == (255, 0, 0)

    def test_total_failure_without_overlay_frame_raises(self) -> None:
        """Test that a CaptureError propagates without an overlay frame."""
        with (
            patch(
                "ocr_tts.ocr_region.capture_region",
                side_effect=CaptureError(["mss/x11: boom"]),
            ),
            patch("ocr_tts.ocr_region._selection_background", None),
            pytest.raises(CaptureError),
        ):
            capture_selected_region(Region(0, 0, 10, 10))


class TestTranslucentOverlay:
    """Tests for the translucent overlay fallback."""

    def test_overlay_translucent_when_no_background(self) -> None:
        """Test that the overlay sets alpha and shows instructions."""
        mock_tk = _install_tkinter_mock()
        try:
            mock_root = mock_tk.Tk.return_value
            mock_canvas = mock_tk.Canvas.return_value
            with (
                patch(
                    "ocr_tts.ocr_region._primary_monitor",
                    return_value=_MONITOR,
                ),
                patch("ocr_tts.ocr_region._capture_background", return_value=None),
            ):
                select_region()
        finally:
            _uninstall_tkinter_mock()

        mock_root.attributes.assert_any_call("-alpha", 0.30)
        mock_canvas.create_text.assert_called_once()

    def test_overlay_opaque_with_background(self) -> None:
        """Test that no alpha is set when a background frame exists."""
        import sys

        mock_tk = _install_tkinter_mock()
        sys.modules["PIL.ImageTk"] = MagicMock()
        try:
            mock_root = mock_tk.Tk.return_value
            with (
                patch(
                    "ocr_tts.ocr_region._primary_monitor",
                    return_value=_MONITOR,
                ),
                patch(
                    "ocr_tts.ocr_region._capture_background",
                    return_value=Image.new("RGB", (10, 10)),
                ),
            ):
                select_region()
        finally:
            sys.modules.pop("PIL.ImageTk", None)
            _uninstall_tkinter_mock()

        alpha_calls = [
            call
            for call in mock_root.attributes.call_args_list
            if call[0][0] == "-alpha"
        ]
        assert alpha_calls == []


class TestIsWayland:
    """Tests for _is_wayland function."""

    def test_not_wayland(self) -> None:
        """Test that no Wayland env vars means not Wayland."""
        with (
            patch.dict(os.environ, {}, clear=True),
        ):
            assert _is_wayland() is False

    def test_wayland_display(self) -> None:
        """Test that WAYLAND_DISPLAY indicates Wayland."""
        with patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}):
            assert _is_wayland() is True

    def test_xdg_session_type(self) -> None:
        """Test that the session type indicates Wayland."""
        with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}):
            assert _is_wayland() is True


class TestExtractText:
    """Tests for extract_text function."""

    def test_extract_text_returns_string(self) -> None:
        """Test that extract_text returns extracted text."""
        mock_image = MagicMock()

        with patch(
            "ocr_tts.ocr_region.pytesseract.image_to_string",
            return_value="Hello World",
        ) as mock_ocr:
            text = extract_text(mock_image)

        assert text == "Hello World"
        mock_ocr.assert_called_once()

    def test_extract_text_with_custom_config(self) -> None:
        """Test that extract_text uses custom config."""
        mock_image = MagicMock()
        config = OCRConfig(lang="fra", tesseract_cmd="/usr/bin/tesseract")

        with patch(
            "ocr_tts.ocr_region.pytesseract.image_to_string",
            return_value="Bonjour",
        ) as mock_ocr:
            text = extract_text(mock_image, config)

        assert text == "Bonjour"
        mock_ocr.assert_called_once_with(mock_image, lang="fra")

    def test_extract_text_strips_whitespace(self) -> None:
        """Test that extract_text strips whitespace."""
        mock_image = MagicMock()

        with patch(
            "ocr_tts.ocr_region.pytesseract.image_to_string",
            return_value="  Hello  \n",
        ):
            text = extract_text(mock_image)

        assert text == "Hello"


# CLI Tests


def test_cli_ocr_region(runner: CliRunner) -> None:
    """Test CLI ocr-region command with mocked dependencies."""
    with (
        patch(
            "ocr_tts.cli.select_region",
            return_value=Region(x=0, y=0, width=100, height=50),
        ),
        patch("ocr_tts.cli.capture_selected_region") as mock_capture,
    ):
        mock_image = MagicMock()
        mock_capture.return_value = mock_image
        with (
            patch("ocr_tts.cli.image_is_blank", return_value=False),
            patch(
                "ocr_tts.cli.extract_text",
                return_value="Extracted Text",
            ),
        ):
            result = runner.invoke(app, ["ocr"])

    assert result.exit_code == 0
    assert "Extracted Text" in result.output


def test_cli_ocr_region_custom_lang(runner: CliRunner) -> None:
    """Test CLI ocr-region with custom language."""
    with (
        patch(
            "ocr_tts.cli.select_region",
            return_value=Region(x=0, y=0, width=100, height=50),
        ),
        patch("ocr_tts.cli.capture_selected_region") as mock_capture,
    ):
        mock_image = MagicMock()
        mock_capture.return_value = mock_image
        with (
            patch("ocr_tts.cli.image_is_blank", return_value=False),
            patch(
                "ocr_tts.cli.extract_text",
                return_value="Bonjour",
            ) as mock_extract,
        ):
            result = runner.invoke(app, ["ocr", "--lang", "fra"])

    assert result.exit_code == 0
    assert "Bonjour" in result.output
    mock_extract.assert_called_once()
    args = mock_extract.call_args
    # config is passed as the second positional argument
    passed_config = args[0][1] if args[0] else args[1].get("config")
    assert passed_config.lang == "fra"


def test_cli_ocr_region_copy_flags(runner: CliRunner) -> None:
    """Test that --copy-text/--copy-image copy via the desktop layer."""
    with (
        patch(
            "ocr_tts.cli.select_region",
            return_value=Region(x=0, y=0, width=100, height=50),
        ),
        patch("ocr_tts.cli.capture_selected_region") as mock_capture,
        patch("ocr_tts.cli.image_is_blank", return_value=False),
        patch("ocr_tts.cli.extract_text", return_value="Clip Text"),
        patch("ocr_tts.cli.copy_text", return_value=True) as mock_copy_text,
        patch("ocr_tts.cli.copy_image", return_value=True) as mock_copy_image,
    ):
        image = Image.new("RGB", (100, 50))
        pixels = image.load()
        assert pixels is not None
        for x in range(0, 100, 2):
            pixels[x, 0] = (255, 255, 255)
        mock_capture.return_value = image
        result = runner.invoke(app, ["ocr", "--copy-text", "--copy-image"])

    assert result.exit_code == 0
    assert "Clip Text" in result.output
    assert "copied to the clipboard" in result.output
    mock_copy_text.assert_called_once_with("Clip Text")
    mock_copy_image.assert_called_once_with(image)


def test_cli_version(runner: CliRunner) -> None:
    """Test CLI --version flag."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "ocr-tts" in result.output


def test_cli_version_short(runner: CliRunner) -> None:
    """Test CLI -V short flag."""
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert "ocr-tts" in result.output


def test_main_entry_point() -> None:
    """Test that __main__.py can be invoked as python -m."""
    from subprocess import run
    from sys import executable

    result = run(
        [executable, "-m", "ocr_tts", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ocr-tts" in result.stdout


def test_main_app_invocation() -> None:
    """Test that __main__.py covers the app() call in-process."""
    from runpy import run_module
    from unittest.mock import patch

    with patch("sys.argv", ["ocr_tts", "--version"]):
        with pytest.raises(SystemExit) as exc_info:
            run_module("ocr_tts.__main__", run_name="__main__")
        assert exc_info.value.code == 0


def test_cli_ocr_region_no_text_detected(runner: CliRunner) -> None:
    """Test CLI ocr-region when OCR returns no text."""
    with (
        patch(
            "ocr_tts.cli.select_region",
            return_value=Region(x=0, y=0, width=100, height=50),
        ),
        patch("ocr_tts.cli.capture_selected_region") as mock_capture,
    ):
        mock_image = MagicMock()
        mock_capture.return_value = mock_image
        with (
            patch("ocr_tts.cli.image_is_blank", return_value=False),
            patch(
                "ocr_tts.cli.extract_text",
                return_value="",
            ),
        ):
            result = runner.invoke(app, ["ocr"])

    assert result.exit_code == 0
    assert "no text detected" in result.output


def test_cli_ocr_region_zero_size_region(runner: CliRunner) -> None:
    """Test CLI ocr-region when selected region has zero size."""
    with patch(
        "ocr_tts.cli.select_region",
        return_value=Region(x=0, y=0, width=0, height=0),
    ):
        result = runner.invoke(app, ["ocr"])

    assert result.exit_code == 0
    assert "No region selected" in result.output
