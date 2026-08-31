"""Tests for the cross-platform clipboard helpers in ocr_tts.desktop."""

import io
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ocr_tts import desktop


def _ok_process(stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    """Build a mocked successful CompletedProcess."""
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def _which_map(mapping: dict[str, str | None]) -> Any:
    """Return a shutil.which replacement driven by a name->path mapping."""
    return lambda name: mapping.get(name)


class TestRunClipTool:
    """Tests for the low-level clipboard tool runner."""

    def test_missing_tool_returns_false(self) -> None:
        """Test that a missing executable quietly returns False."""
        with (
            patch("ocr_tts.desktop.shutil.which", return_value=None),
            patch("ocr_tts.desktop.subprocess.run") as mock_run,
        ):
            assert desktop._run_clip_tool(["wl-copy"], b"data") is False

        mock_run.assert_not_called()

    def test_failing_tool_returns_false_and_logs(self) -> None:
        """Test that a non-zero exit returns False."""
        proc = MagicMock()
        proc.returncode = 1
        proc.stderr = b"boom"
        with (
            patch(
                "ocr_tts.desktop.shutil.which",
                side_effect=_which_map({"xclip": "/usr/bin/xclip"}),
            ),
            patch("ocr_tts.desktop.subprocess.run", return_value=proc),
        ):
            assert desktop._run_clip_tool(["xclip"], None) is False


class TestCopyText:
    """Tests for copy_text."""

    def test_wayland_prefers_wl_copy(self) -> None:
        """Test that wl-copy is used first on Wayland."""
        with (
            patch("ocr_tts.desktop._is_wayland", return_value=True),
            patch(
                "ocr_tts.desktop.shutil.which",
                side_effect=_which_map(
                    {"/usr/bin/wl-copy": "", "wl-copy": "/usr/bin/wl-copy"}
                ),
            ),
            patch(
                "ocr_tts.desktop.subprocess.run",
                return_value=_ok_process(),
            ) as mock_run,
        ):
            assert desktop.copy_text("hello") is True

        argv = mock_run.call_args[0][0]
        assert argv == ["/usr/bin/wl-copy"]
        assert mock_run.call_args[1]["input"] == b"hello"

    def test_falls_back_to_xclip_when_no_wl_copy(self) -> None:
        """Test that xclip is tried when wl-copy is not installed."""
        with (
            patch("ocr_tts.desktop._is_wayland", return_value=True),
            patch(
                "ocr_tts.desktop.shutil.which",
                side_effect=_which_map({"xclip": "/usr/bin/xclip"}),
            ),
            patch(
                "ocr_tts.desktop.subprocess.run",
                return_value=_ok_process(),
            ) as mock_run,
        ):
            assert desktop.copy_text("hello") is True

        argv = mock_run.call_args[0][0]
        assert argv[:2] == ["/usr/bin/xclip", "-selection"]

    def test_returns_false_without_backends(self) -> None:
        """Test that copy_text fails cleanly when no tool exists."""
        with (
            patch("ocr_tts.desktop._is_wayland", return_value=False),
            patch("ocr_tts.desktop.shutil.which", return_value=None),
        ):
            assert desktop.copy_text("hello") is False


class TestCopyImage:
    """Tests for copy_image."""

    def _image(self) -> Image.Image:
        """Build a small textured test image."""
        image = Image.new("RGB", (20, 10), color=(0, 0, 0))
        pixels = image.load()
        assert pixels is not None
        for x in range(0, 20, 2):
            pixels[x, 0] = (255, 255, 255)
        return image

    def test_linux_uses_xclip_with_png_type(self) -> None:
        """Test that xclip is invoked with the image/png target."""
        png_buffer = io.BytesIO()
        self._image().save(png_buffer, format="PNG")
        expected = png_buffer.getvalue()

        with (
            patch("ocr_tts.desktop._is_wayland", return_value=False),
            patch(
                "ocr_tts.desktop.shutil.which",
                side_effect=_which_map({"xclip": "/usr/bin/xclip"}),
            ),
            patch(
                "ocr_tts.desktop.subprocess.run",
                return_value=_ok_process(),
            ) as mock_run,
        ):
            assert desktop.copy_image(self._image()) is True

        argv = mock_run.call_args[0][0]
        assert argv == [
            "/usr/bin/xclip",
            "-selection",
            "clipboard",
            "-t",
            "image/png",
        ]
        assert mock_run.call_args[1]["input"] == expected

    def test_windows_stages_temp_file_for_powershell(self) -> None:
        """Test that Windows copies via PowerShell using a temp PNG."""
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("ocr_tts.desktop.sys.platform", "win32"),
            patch(
                "ocr_tts.desktop.tempfile.mkstemp",
                return_value=(9, "staged-clip.png"),
            ),
            patch("ocr_tts.desktop.os.fdopen") as mock_fdopen,
            patch("ocr_tts.desktop.os.unlink") as mock_unlink,
            patch(
                "ocr_tts.desktop.shutil.which",
                side_effect=_which_map({"powershell": "/usr/bin/powershell"}),
            ),
            patch(
                "ocr_tts.desktop.subprocess.run",
                return_value=_ok_process(),
            ) as mock_run,
        ):
            mock_fdopen.return_value.__enter__ = MagicMock()
            assert desktop.copy_image(self._image()) is True

        argv = mock_run.call_args[0][0]
        assert argv[0] == "/usr/bin/powershell"
        assert "-STA" in argv
        script = argv[-1]
        assert "Clipboard]::SetImage" in script
        assert "staged-clip.png" in script
        mock_unlink.assert_called_once()

    def test_macos_uses_osascript_with_temp_file(self) -> None:
        """Test that macOS copies via osascript reading a temp PNG."""
        with (
            patch("ocr_tts.desktop.sys.platform", "darwin"),
            patch(
                "ocr_tts.desktop.tempfile.mkstemp",
                return_value=(9, "staged-clip.png"),
            ),
            patch("ocr_tts.desktop.os.fdopen") as mock_fdopen,
            patch("ocr_tts.desktop.os.unlink") as mock_unlink,
            patch(
                "ocr_tts.desktop.shutil.which",
                side_effect=_which_map({"osascript": "/usr/bin/osascript"}),
            ),
            patch(
                "ocr_tts.desktop.subprocess.run",
                return_value=_ok_process(),
            ) as mock_run,
        ):
            mock_fdopen.return_value.__enter__ = MagicMock()
            assert desktop.copy_image(self._image()) is True

        argv = mock_run.call_args[0][0]
        assert argv[:2] == ["/usr/bin/osascript", "-e"]
        assert "«class PNGf»" in argv[2]
        assert "staged-clip.png" in argv[2]
        mock_unlink.assert_called_once()

    def test_returns_false_without_backends(self) -> None:
        """Test that copy_image fails cleanly when no tool exists."""
        with (
            patch("ocr_tts.desktop._is_wayland", return_value=False),
            patch("ocr_tts.desktop.shutil.which", return_value=None),
        ):
            assert desktop.copy_image(self._image()) is False


class TestTextBackendsPerPlatform:
    """Tests for backend selection by platform."""

    def test_darwin_uses_pbcopy(self) -> None:
        """Test that pbcopy is the only macOS text backend."""
        with patch("ocr_tts.desktop.sys.platform", "darwin"):
            backends = desktop._text_backends()

        assert backends == [["pbcopy"]]

    def test_windows_uses_clip_and_powershell(self) -> None:
        """Test that clip.exe comes first, then PowerShell."""
        with patch("ocr_tts.desktop.sys.platform", "win32"):
            backends = desktop._text_backends()

        assert backends[0][0] == "clip"
        assert backends[1][0] == "powershell"

    def test_x11_uses_xclip_then_xsel(self) -> None:
        """Test that X11 sessions try xclip before xsel."""
        with patch("ocr_tts.desktop._is_wayland", return_value=False):
            backends = desktop._text_backends()

        assert [b[0] for b in backends] == ["xclip", "xsel"]


class TestPipeWireCapture:
    """Tests for the PipeWire/GStreamer capture backend."""

    def _png_bytes(self) -> bytes:
        """Encode a small textured PNG into bytes."""
        image = Image.new("RGB", (40, 30), color=(0, 0, 0))
        pixels = image.load()
        assert pixels is not None
        for x in range(0, 40, 2):
            pixels[x, 0] = (255, 255, 255)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_pipewire_video_nodes_prefers_gamescope(self) -> None:
        """Test that pw-dump parsing lists gamescope nodes first."""
        dump = [
            {
                "id": 40,
                "info": {"props": {"media.class": "Video/Source"}},
            },
            {
                "id": 42,
                "info": {
                    "props": {
                        "media.class": "Video/Source",
                        "node.name": "gamescope",
                    }
                },
            },
            {
                "id": 43,
                "info": {"props": {"media.class": "Audio/Source"}},
            },
            {
                "id": 44,
                "info": {"props": {"media.class": "Video/Sink"}},
            },
        ]

        def fake_run(*_args: Any, **_kwargs: Any) -> MagicMock:
            proc = MagicMock()
            proc.returncode = 0
            import json as json_mod

            proc.stdout = json_mod.dumps(dump).encode()
            return proc

        with (
            patch(
                "ocr_tts.desktop.shutil.which",
                side_effect=_which_map({"pw-dump": "/usr/bin/pw-dump"}),
            ),
            patch("ocr_tts.desktop.subprocess.run", side_effect=fake_run),
        ):
            nodes = desktop._pipewire_video_nodes()

        assert nodes[0] == "42"
        assert nodes[1] == "40"

    def test_pipewire_capture_with_env_node(self) -> None:
        """Test that OCR_TTS_PIPEWIRE_NODE pins the node and path prop."""

        def fake_run(cmd: list[str], **_kwargs: Any) -> MagicMock:
            if "path=99" in cmd:
                proc = MagicMock()
                proc.returncode = 0
                proc.stderr = b""

                with open(cmd[-1].split("=", 1)[1], "wb") as handle:
                    handle.write(self._png_bytes())
                return proc
            proc = MagicMock()
            proc.returncode = 1
            proc.stderr = b"nope"
            return proc

        with (
            patch.dict(os.environ, {"OCR_TTS_PIPEWIRE_NODE": "99"}),
            patch(
                "ocr_tts.desktop.shutil.which",
                side_effect=_which_map({"gst-launch-1.0": "/usr/bin/gst-launch-1.0"}),
            ),
            patch("ocr_tts.desktop.subprocess.run", side_effect=fake_run) as mock_run,
            # Pin monitor bounds to the fake frame so the region crop is
            # deterministic on hosts with a real display attached.
            patch(
                "ocr_tts.desktop._primary_monitor",
                return_value={"left": 0, "top": 0, "width": 40, "height": 30},
            ),
        ):
            image = desktop._capture_region_pipewire(0, 0, 20, 15)

        assert image.size == (20, 15)
        first_cmd = mock_run.call_args_list[0][0][0]
        assert first_cmd[:4] == [
            "/usr/bin/gst-launch-1.0",
            "-q",
            "pipewiresrc",
            "path=99",
        ]
        assert first_cmd[-2:] == ["filesink", mock_run.call_args_list[0][0][0][-1]]

    def test_pipewire_capture_falls_back_to_target_object(self) -> None:
        """Test the target-object property when path= is unsupported."""

        def fake_run(cmd: list[str], **_kwargs: Any) -> MagicMock:
            if any(arg.startswith("target-object=") for arg in cmd):
                with open(cmd[-1].split("=", 1)[1], "wb") as handle:
                    handle.write(self._png_bytes())
                proc = MagicMock()
                proc.returncode = 0
                proc.stderr = b""
                return proc
            proc = MagicMock()
            proc.returncode = 1
            proc.stderr = b"no property named path"
            return proc

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ocr_tts.desktop.shutil.which",
                side_effect=_which_map({"gst-launch-1.0": "/usr/bin/gst-launch-1.0"}),
            ),
            patch(
                "ocr_tts.desktop._pipewire_video_nodes",
                return_value=["7"],
            ),
            patch("ocr_tts.desktop.subprocess.run", side_effect=fake_run),
            patch(
                "ocr_tts.desktop._primary_monitor",
                return_value={"left": 0, "top": 0, "width": 40, "height": 30},
            ),
        ):
            image = desktop._capture_region_pipewire(0, 0, 20, 15)

        assert image.size == (20, 15)

    def test_pipewire_capture_requires_gstreamer(self) -> None:
        """Test a clean failure when gst-launch-1.0 is missing."""
        with (
            patch("ocr_tts.desktop.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match=r"gst-launch-1\.0"),
        ):
            desktop._capture_region_pipewire(0, 0, 10, 10)
