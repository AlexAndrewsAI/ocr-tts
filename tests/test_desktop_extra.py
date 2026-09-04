"""Additional coverage tests for the desktop platform services layer."""

import io
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ocr_tts import desktop


def _blank_shot(width: int = 10, height: int = 10) -> MagicMock:
    """Build a mocked mss screenshot of solid black."""
    shot = MagicMock()
    shot.size = (width, height)
    shot.rgb = bytes(width * height * 3)
    return shot


def _mock_sct(shot: MagicMock) -> MagicMock:
    """Wrap a screenshot in a mocked mss context manager."""
    sct = MagicMock()
    sct.__enter__ = MagicMock(return_value=sct)
    sct.__exit__ = MagicMock(return_value=False)
    sct.grab.return_value = shot
    return sct


class TestCaptureRegionX11Failures:
    """Tests for the X11 backend's retry/raise paths."""

    def test_all_blank_frames_raise(self) -> None:
        """Two consecutive blank grabs raise a blank-frame RuntimeError."""
        with (
            patch("ocr_tts.desktop.time.sleep"),
            patch("ocr_tts.desktop.mss.MSS", return_value=_mock_sct(_blank_shot())),
            pytest.raises(RuntimeError, match="blank frames"),
        ):
            desktop._capture_region_x11(0, 0, 10, 10)

    def test_persistent_failure_raises_with_error(self) -> None:
        """Repeated grab failures raise with the underlying error."""
        with (
            patch("ocr_tts.desktop.time.sleep"),
            patch(
                "ocr_tts.desktop.mss.MSS",
                side_effect=RuntimeError("no display"),
            ),
            pytest.raises(RuntimeError, match="X11 screen capture failed"),
        ):
            desktop._capture_region_x11(0, 0, 10, 10)


class TestCropFallback:
    """Tests for the crop helper's monitor-probe fallback."""

    def test_probe_failure_uses_frame_dimensions(self) -> None:
        """When the monitor probe fails the frame itself is the reference."""
        full = Image.new("RGB", (100, 80), color=(1, 2, 3))
        with patch.object(desktop, "_primary_monitor", side_effect=RuntimeError):
            image = desktop._crop_fullscreen_to_region(full, 10, 20, 50, 40)
        assert image.size == (50, 40)
        assert image.getpixel((0, 0)) == (1, 2, 3)


class TestCropAspectDiagnostics:
    """Tests for aspect-ratio mismatch warnings in the crop helper."""

    def test_aspect_mismatch_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A frame whose aspect differs from the logical screen warns."""
        full = Image.new("RGB", (1920, 1080))
        monitor = {"left": 0, "top": 0, "width": 1920, "height": 1200}
        with (
            patch.object(desktop, "_primary_monitor", return_value=monitor),
            caplog.at_level(logging.WARNING, logger="ocr_tts.desktop"),
        ):
            desktop._crop_fullscreen_to_region(full, 0, 0, 10, 10)
        assert "aspect ratio" in caplog.text

    def test_matching_aspect_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning is emitted when the aspect ratios agree."""
        full = Image.new("RGB", (3840, 2160))
        monitor = {"left": 0, "top": 0, "width": 1920, "height": 1080}
        with (
            patch.object(desktop, "_primary_monitor", return_value=monitor),
            caplog.at_level(logging.WARNING, logger="ocr_tts.desktop"),
        ):
            desktop._crop_fullscreen_to_region(full, 0, 0, 10, 10)
        assert "aspect ratio" not in caplog.text


class TestGrimBackend:
    """Tests for the grim full-frame grab and scaled crop."""

    def _png_frame(self, size: tuple[int, int]) -> bytes:
        """Encode a solid non-black PNG frame of the given size."""
        frame = Image.new("RGB", size, color=(12, 34, 56))
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_full_grab_is_cropped_with_scaled_coordinates(self) -> None:
        """A higher-res output frame is cropped via logical-to-pixel map."""
        proc = MagicMock(returncode=0, stdout=self._png_frame((60, 60)), stderr=b"")
        monitor = {"left": 0, "top": 0, "width": 30, "height": 30}
        with (
            patch.object(desktop, "_require_tool", return_value="/usr/bin/grim"),
            patch.object(desktop.subprocess, "run", return_value=proc) as mock_run,
            patch.object(desktop, "_primary_monitor", return_value=monitor),
        ):
            image = desktop._capture_region_grim(10, 12, 10, 10)

        cmd = mock_run.call_args[0][0]
        assert cmd == ["/usr/bin/grim", "-"]
        # Scale factor 2: logical (10,12) maps to frame (20,24).
        assert image.size == (20, 20)
        assert image.getpixel((0, 0)) == (12, 34, 56)


class TestCaptureRegionLogging:
    """Tests for backend success logging in the capture chain."""

    def test_success_logs_backend_and_size(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The winning backend name and image size are logged."""
        frame = Image.new("RGB", (40, 20))
        pixels = frame.load()
        assert pixels is not None
        for x in range(40):
            for y in range(20):
                pixels[x, y] = ((x * 7) % 256, (y * 13) % 256, 30)
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        proc = MagicMock(returncode=0, stdout=buffer.getvalue(), stderr=b"")
        with (
            patch.dict(os.environ, {"OCR_TTS_CAPTURE_COMMAND": "fake-tool {w}x{h}"}),
            patch.object(desktop.subprocess, "run", return_value=proc),
            caplog.at_level(logging.INFO, logger="ocr_tts.desktop"),
        ):
            image = desktop.capture_region(0, 0, 10, 10)

        assert image.size == (40, 20)
        assert "custom command capture succeeded (40x20)" in caplog.text


class TestRunToolCapture:
    """Tests for external capture-tool execution."""

    def test_timeout_raises(self) -> None:
        """A tool that hangs raises a timeout error."""
        with (
            patch(
                "ocr_tts.desktop.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["grim"], timeout=1),
            ),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            desktop._run_tool_capture(["grim", "-"])

    def test_nonzero_exit_raises(self) -> None:
        """A failing tool surfaces its stderr."""
        proc = MagicMock(returncode=2, stdout=b"", stderr=b"boom")
        with (
            patch("ocr_tts.desktop.subprocess.run", return_value=proc),
            pytest.raises(RuntimeError, match="boom"),
        ):
            desktop._run_tool_capture(["grim", "-"])

    def test_success_decodes_png(self) -> None:
        """Valid PNG on stdout is decoded to an RGB image."""
        buf = Image.new("RGB", (8, 6), color=(7, 8, 9))
        import io

        png = io.BytesIO()
        buf.save(png, format="PNG")
        proc = MagicMock(returncode=0, stdout=png.getvalue(), stderr=b"")
        with patch("ocr_tts.desktop.subprocess.run", return_value=proc):
            image = desktop._run_tool_capture(["grim", "-"])
        assert image.size == (8, 6)
        assert image.getpixel((0, 0)) == (7, 8, 9)


class TestMaimBackend:
    """Tests for the maim capture backend."""

    def test_builds_geometry_argument(self) -> None:
        """Maim receives an explicit geometry string."""
        image = Image.new("RGB", (30, 20))
        with (
            patch.object(desktop, "_require_tool", return_value="/usr/bin/maim"),
            patch.object(desktop, "_run_tool_capture", return_value=image) as run,
        ):
            result = desktop._capture_region_maim(5, 6, 30, 20)
        assert result is image
        cmd = run.call_args[0][0]
        assert cmd[0] == "/usr/bin/maim"
        assert "30x20+5+6" in cmd


class TestCaptureViaFile:
    """Tests for file-based capture helpers."""

    @staticmethod
    def _run_writes_png(cmd: list[str], **_kwargs: Any) -> MagicMock:
        """Fake subprocess that writes a PNG to the -o argument."""
        png_arg = next(a for a in cmd if a.endswith(".png"))
        Image.new("RGB", (100, 80), color=(9, 9, 9)).save(png_arg)
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    def test_success_crops_region(self) -> None:
        """A written full-screen file is cropped to the region."""
        with (
            patch.object(desktop, "_require_tool", return_value="/usr/bin/scrot"),
            patch(
                "ocr_tts.desktop.subprocess.run",
                side_effect=self._run_writes_png,
            ),
            patch.object(
                desktop,
                "_primary_monitor",
                return_value={"left": 0, "top": 0, "width": 100, "height": 80},
            ),
        ):
            image = desktop._capture_via_file(
                "scrot", lambda p: ["-o", p], 10, 20, 30, 40
            )
        assert image.size == (30, 40)

    def test_timeout_raises(self) -> None:
        """A hanging tool raises a timeout error."""
        with (
            patch.object(desktop, "_require_tool", return_value="/usr/bin/scrot"),
            patch(
                "ocr_tts.desktop.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["scrot"], timeout=1),
            ),
            pytest.raises(RuntimeError, match="scrot timed out"),
        ):
            desktop._capture_via_file("scrot", lambda p: ["-o", p], 0, 0, 5, 5)

    def test_failure_raises(self) -> None:
        """A non-zero exit raises with the tool's stderr."""
        proc = MagicMock(returncode=3, stderr=b"cannot open display")
        with (
            patch.object(desktop, "_require_tool", return_value="/usr/bin/scrot"),
            patch("ocr_tts.desktop.subprocess.run", return_value=proc),
            pytest.raises(RuntimeError, match="scrot failed"),
        ):
            desktop._capture_via_file("scrot", lambda p: ["-o", p], 0, 0, 5, 5)


class TestPipewireNodes:
    """Tests for PipeWire node discovery."""

    def test_missing_pw_dump_returns_empty(self) -> None:
        """Without pw-dump no nodes are reported."""
        with patch("ocr_tts.desktop.shutil.which", return_value=None):
            assert desktop._pipewire_video_nodes() == []

    def test_pw_dump_timeout_returns_empty(self) -> None:
        """pw-dump timeouts are swallowed."""
        with (
            patch("ocr_tts.desktop.shutil.which", return_value="/usr/bin/pw-dump"),
            patch(
                "ocr_tts.desktop.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["pw-dump"], timeout=1),
            ),
        ):
            assert desktop._pipewire_video_nodes() == []

    def test_pw_dump_bad_json_returns_empty(self) -> None:
        """Unparseable pw-dump output yields no nodes."""
        proc = MagicMock(returncode=0, stdout=b"not json")
        with (
            patch("ocr_tts.desktop.shutil.which", return_value="/usr/bin/pw-dump"),
            patch("ocr_tts.desktop.subprocess.run", return_value=proc),
        ):
            assert desktop._pipewire_video_nodes() == []

    def test_pw_dump_failure_returncode(self) -> None:
        """A failing pw-dump yields no nodes."""
        proc = MagicMock(returncode=1, stdout=b"[]")
        with (
            patch("ocr_tts.desktop.shutil.which", return_value="/usr/bin/pw-dump"),
            patch("ocr_tts.desktop.subprocess.run", return_value=proc),
        ):
            assert desktop._pipewire_video_nodes() == []

    def test_entries_filtering_and_gamescope_priority(self) -> None:
        """Sinks and invalid entries are skipped; gamescope nodes come first."""
        import json

        entries = [
            {"id": "55", "info": {"props": {"media.class": "Video/Source"}}},
            {
                "id": "56",
                "info": {
                    "props": {
                        "media.class": "Video/Source",
                        "node.name": "gamescope-output",
                    }
                },
            },
            {"id": "", "info": {"props": {"media.class": "Video/Source"}}},
            {"id": "57", "info": {"props": {"media.class": "Video/Sink"}}},
            {"id": "58", "info": {"props": {"media.class": "Audio/Source"}}},
            {"info": {}},
        ]
        proc = MagicMock(returncode=0, stdout=json.dumps(entries).encode())
        with (
            patch("ocr_tts.desktop.shutil.which", return_value="/usr/bin/pw-dump"),
            patch("ocr_tts.desktop.subprocess.run", return_value=proc),
        ):
            nodes = desktop._pipewire_video_nodes()
        assert nodes == ["56", "55"]

    def test_non_list_payload_returns_empty(self) -> None:
        """A JSON object payload yields no nodes."""
        import json

        proc = MagicMock(returncode=0, stdout=json.dumps({"foo": 1}).encode())
        with (
            patch("ocr_tts.desktop.shutil.which", return_value="/usr/bin/pw-dump"),
            patch("ocr_tts.desktop.subprocess.run", return_value=proc),
        ):
            assert desktop._pipewire_video_nodes() == []


class TestPipewireCapture:
    """Tests for the GStreamer PipeWire capture path."""

    def test_no_nodes_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without any video node the capture fails fast."""
        monkeypatch.delenv("OCR_TTS_PIPEWIRE_NODE", raising=False)

        def which(name: str) -> str | None:
            return "/usr/bin/gst-launch-1.0" if name == "gst-launch-1.0" else None

        with (
            patch("ocr_tts.desktop.shutil.which", side_effect=which),
            pytest.raises(RuntimeError, match="no PipeWire video node"),
        ):
            desktop._capture_region_pipewire(0, 0, 10, 10)

    def test_gst_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hanging gst-launch raises a timeout error."""
        monkeypatch.setenv("OCR_TTS_PIPEWIRE_NODE", "99")
        with (
            patch("ocr_tts.desktop.shutil.which", return_value="/usr/bin/gst"),
            patch(
                "ocr_tts.desktop.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["gst"], timeout=1),
            ),
            pytest.raises(RuntimeError, match=r"gst-launch-1\.0 timed out"),
        ):
            desktop._capture_region_pipewire(0, 0, 10, 10)

    def test_all_attempts_fail_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exhausting all node/property attempts raises a summary error."""
        monkeypatch.setenv("OCR_TTS_PIPEWIRE_NODE", "99")
        proc = MagicMock(returncode=1, stderr=b"link failed")
        with (
            patch("ocr_tts.desktop.shutil.which", return_value="/usr/bin/gst"),
            patch("ocr_tts.desktop.subprocess.run", return_value=proc),
            pytest.raises(RuntimeError, match="PipeWire capture failed"),
        ):
            desktop._capture_region_pipewire(0, 0, 10, 10)


class TestClipToolErrors:
    """Tests for clipboard tool failure handling."""

    def test_oserror_is_swallowed(self) -> None:
        """An OSError from the tool reports failure without raising."""
        with (
            patch("ocr_tts.desktop.shutil.which", return_value="/usr/bin/xclip"),
            patch("ocr_tts.desktop.subprocess.run", side_effect=OSError("gone")),
        ):
            assert desktop._run_clip_tool(["xclip"], b"data") is False


class TestImageBackendsPlatforms:
    """Tests for platform-specific image clipboard backends."""

    def test_darwin_requires_png_path(self, tmp_path: Path) -> None:
        """MacOS backends need a staged PNG file."""
        with patch.object(sys, "platform", "darwin"):
            with pytest.raises(RuntimeError, match="PNG file"):
                desktop._image_backends(None)
            cmds = desktop._image_backends(str(tmp_path / "x.png"))
            assert cmds[0][0] == "osascript"

    def test_win32_requires_png_path(self, tmp_path: Path) -> None:
        """Windows backends need a staged PNG file."""
        with patch.object(sys, "platform", "win32"):
            with pytest.raises(RuntimeError, match="PNG file"):
                desktop._image_backends(None)
            cmds = desktop._image_backends(str(tmp_path / "x.png"))
            assert cmds[0][0] == "powershell"


class TestImageBackendsLinuxWayland:
    """Tests for the Linux/Wayland image clipboard backend list."""

    def test_wayland_adds_wl_copy(self) -> None:
        """Wayland sessions try wl-copy before xclip."""
        with (
            patch.object(desktop, "_is_wayland", return_value=True),
            patch.object(sys, "platform", "linux"),
        ):
            cmds = desktop._image_backends(None)
        assert cmds[0][0] == "wl-copy"
        assert cmds[1][0] == "xclip"
