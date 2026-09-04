"""Platform services layer: screen capture and clipboard access.

Consolidates every OS-dependent operation used by the OCR pipeline so
the rest of the codebase stays platform-agnostic:

* :func:`capture_region` -- screen capture through a chain of backends
  covering X11, Wayland compositors, nested compositors such as
  gamescope (via PipeWire, which is the only readback path that sees
  real game content there), and external helper tools.
* :func:`copy_text` / :func:`copy_image` -- best-effort clipboard
  writes using whichever backend is available (wl-copy, xclip, xsel,
  pbcopy, clip.exe, PowerShell, osascript).
"""

import asyncio
import contextlib
import io
import json
import logging
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from urllib.parse import unquote, urlparse

import mss
from PIL import Image, ImageStat

logger = logging.getLogger(__name__)

_capture_tool_timeout = 15.0

_portal_unavailable = False


class CaptureError(RuntimeError):
    """Raised when every screen-capture backend failed."""

    def __init__(self, errors: list[str]) -> None:
        """Build the error from the per-backend failure messages."""
        self.errors = errors
        detail = "; ".join(errors) if errors else "no backends available"
        super().__init__(f"screen capture failed on all backends: {detail}")


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------


def image_is_blank(image: Image.Image, threshold: float = 5.0) -> bool:
    """Return True if the image is essentially a single uniform color.

    Used to detect failed captures that silently return a black or
    otherwise empty frame.

    Args:
        image: The image to inspect.
        threshold: Maximum luminance standard deviation considered blank.

    Returns:
        True when the image has almost no luminance variation.

    """
    stat = ImageStat.Stat(image.convert("L"))
    return float(stat.stddev[0]) < threshold


def _is_wayland() -> bool:
    """Return True when running under a Wayland compositor."""
    return bool(os.environ.get("WAYLAND_DISPLAY")) or (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    )


def _monitor_bounds_tkinter() -> dict[str, int]:
    """Return the root screen bounds using tkinter.

    Works on X11 and XWayland (including gamescope) even when mss
    cannot query the display, e.g. on native Wayland sessions.

    Returns:
        An mss-style monitor bounds dict anchored at (0, 0).

    Raises:
        RuntimeError: If tkinter or the display is unavailable.

    """
    import tkinter as tk

    try:
        probe = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(f"tkinter screen probe failed: {exc}") from exc
    probe.withdraw()
    try:
        return {
            "left": 0,
            "top": 0,
            "width": int(probe.winfo_screenwidth()),
            "height": int(probe.winfo_screenheight()),
        }
    finally:
        probe.destroy()


def _primary_monitor() -> dict[str, int]:
    """Return the primary monitor bounds as a mss-style dict."""
    try:
        with mss.MSS() as sct:
            monitor = sct.monitors[1]
    except Exception as exc:
        logger.info("mss monitor lookup failed (%s); probing via tkinter instead", exc)
        return _monitor_bounds_tkinter()
    return {
        "left": int(monitor["left"]),
        "top": int(monitor["top"]),
        "width": int(monitor["width"]),
        "height": int(monitor["height"]),
    }


def _capture_region_x11(x: int, y: int, w: int, h: int) -> Image.Image:
    """Capture a screen region using the mss/X11 backend.

    A grab that returns a blank frame is retried once after a short
    pause; this covers the transient window between tearing down the
    selection overlay and the compositor repainting the screen.

    Args:
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        w: Width of the region in pixels.
        h: Height of the region in pixels.

    Returns:
        An RGB PIL Image of the captured screen region.

    Raises:
        RuntimeError: If every capture attempt fails.

    """
    monitor = {"left": x, "top": y, "width": w, "height": h}
    image: Image.Image | None = None
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with mss.MSS() as sct:
                shot = sct.grab(monitor)
                image = Image.frombytes("RGB", shot.size, shot.rgb)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "X11 screen capture failed on attempt %d: %s",
                attempt + 1,
                exc,
            )
            time.sleep(0.3)
            continue
        if not image_is_blank(image):
            return image
        logger.warning(
            "X11 screen capture returned a blank frame on attempt %d", attempt + 1
        )
        time.sleep(0.3)
    if image is None:
        raise RuntimeError(f"X11 screen capture failed: {last_error}")
    raise RuntimeError("X11 screen capture returned only blank frames")


def _portal_screenshot(timeout: float = 10.0) -> Image.Image:
    """Capture the full screen via the xdg-desktop-portal Screenshot API.

    Args:
        timeout: Maximum seconds to wait for the portal reply/response.

    Returns:
        A PIL Image of the full screen.

    Raises:
        RuntimeError: If the portal is unavailable or the capture fails.

    """
    try:
        from dbus_fast import BusType, Message, MessageType, Variant
        from dbus_fast.aio import MessageBus
    except ImportError as exc:
        raise RuntimeError(
            "dbus-fast is required for Wayland screen capture. "
            "Install dbus-fast to use this feature."
        ) from exc

    async def _capture() -> Image.Image:
        bus = await MessageBus(bus_type=BusType.SESSION).connect()
        token = f"ocr_region_{os.getpid()}_{secrets.token_hex(4)}"
        call = Message(
            destination="org.freedesktop.portal.Desktop",
            path="/org/freedesktop/portal/desktop",
            interface="org.freedesktop.portal.Screenshot",
            member="Screenshot",
            signature="sa{sv}",
            body=[
                "",
                {
                    "handle_token": Variant("s", token),
                    "interactive": Variant("b", False),
                },
            ],
        )
        try:
            reply = await asyncio.wait_for(bus.call(call), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Timed out calling the screenshot portal") from exc
        if reply.message_type == MessageType.ERROR:
            raise RuntimeError(
                f"Portal Screenshot call failed: {reply.error_name or reply.body}"
            )
        handle = str(reply.body[0])
        future: asyncio.Future[Message] = asyncio.get_running_loop().create_future()

        def on_response(msg: Message) -> None:
            """Complete the future when the portal emits its Response."""
            if (
                msg.interface == "org.freedesktop.portal.Request"
                and msg.member == "Response"
                and str(msg.path) == handle
                and not future.done()
            ):
                future.set_result(msg)

        bus.add_message_handler(on_response)
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Timed out waiting for the portal screenshot") from exc
        finally:
            bus.remove_message_handler(on_response)

        if response.body[0] != 0:
            raise RuntimeError(
                f"Portal screenshot failed with response code {response.body[0]}"
            )
        results = response.body[1]
        uri = results["uri"].value
        image = Image.open(unquote(urlparse(uri).path))
        image.load()
        return image

    return asyncio.run(_capture())


def _crop_fullscreen_to_region(
    full: Image.Image, x: int, y: int, w: int, h: int
) -> Image.Image:
    """Crop a full-screen capture down to the requested region.

    The region coordinates are given in logical screen coordinates
    (what tkinter/X11 report), while the captured frame may have a
    different pixel resolution -- e.g. a gamescope PipeWire frame is
    captured at output resolution which can differ from the XWayland
    logical size. The region is therefore scaled to match the frame.

    Args:
        full: Full-screen PIL Image.
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        w: Width of the region in pixels.
        h: Height of the region in pixels.

    Returns:
        A PIL Image of the requested region.

    """
    try:
        monitor = _primary_monitor()
        offset_x = int(monitor["left"])
        offset_y = int(monitor["top"])
        logical_w = max(1, int(monitor["width"]))
        logical_h = max(1, int(monitor["height"]))
    except Exception:
        offset_x = 0
        offset_y = 0
        logical_w = full.width
        logical_h = full.height
    frame_ratio = full.width / max(1, full.height)
    logical_ratio = logical_w / max(1, logical_h)
    if abs(frame_ratio - logical_ratio) > 0.02 * logical_ratio:
        logger.warning(
            "Capture frame %dx%d differs in aspect ratio from the logical "
            "screen %dx%d; the mapped crop may be offset",
            full.width,
            full.height,
            logical_w,
            logical_h,
        )
    scale_x = full.width / logical_w
    scale_y = full.height / logical_h
    left_f = (x - offset_x) * scale_x
    top_f = (y - offset_y) * scale_y
    right_f = left_f + w * scale_x
    bottom_f = top_f + h * scale_y
    width, height = full.size
    left = max(0, min(int(left_f), width))
    top = max(0, min(int(top_f), height))
    right = max(left, min(int(right_f), width))
    bottom = max(top, min(int(bottom_f), height))
    return full.crop((left, top, right, bottom))


def _capture_region_wayland(x: int, y: int, w: int, h: int) -> Image.Image:
    """Capture a screen region using the xdg-desktop-portal screenshot.

    The portal returns the full screen; the requested region is
    cropped out using the primary monitor origin.

    Args:
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        w: Width of the region in pixels.
        h: Height of the region in pixels.

    Returns:
        A PIL Image of the captured screen region.

    """
    full = _portal_screenshot()
    return _crop_fullscreen_to_region(full, x, y, w, h)


def _decode_png_bytes(data: bytes) -> Image.Image:
    """Decode PNG bytes into an RGB PIL Image."""
    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def _run_tool_capture(cmd: list[str]) -> Image.Image:
    """Run an external capture command that writes a PNG to stdout.

    Args:
        cmd: The command argv to execute.

    Returns:
        An RGB PIL Image produced by the tool.

    Raises:
        RuntimeError: If the command fails or produces no output.

    """
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            timeout=_capture_tool_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{cmd[0]} timed out") from exc
    if proc.returncode != 0 or not proc.stdout:
        stderr = proc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"{cmd[0]} failed (rc={proc.returncode}): {stderr}")
    return _decode_png_bytes(proc.stdout)


def _require_tool(name: str) -> str:
    """Return the path of an external tool or raise RuntimeError."""
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"'{name}' executable not found on PATH")
    return path


def _capture_region_grim(x: int, y: int, w: int, h: int) -> Image.Image:
    """Capture a region with grim (wlroots/gamescope screencopy).

    The full compositor output is grabbed and then cropped with the
    same logical-to-pixel mapping as every other full-frame backend,
    so compositors whose output resolution differs from the logical
    screen size are handled correctly.

    Args:
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        w: Width of the region in pixels.
        h: Height of the region in pixels.

    Returns:
        A cropped RGB PIL Image of the requested region.

    Raises:
        RuntimeError: If grim is missing or fails.

    """
    exe = _require_tool("grim")
    full = _run_tool_capture([exe, "-"])
    return _crop_fullscreen_to_region(full, x, y, w, h)


def _capture_region_maim(x: int, y: int, w: int, h: int) -> Image.Image:
    """Capture a region with maim (X11)."""
    exe = _require_tool("maim")
    return _run_tool_capture([exe, "-g", f"{w}x{h}+{x}+{y}"])


def _capture_via_file(
    name: str, build_cmd: Callable[[str], list[str]], x: int, y: int, w: int, h: int
) -> Image.Image:
    """Capture a full screen via a tool writing a file, then crop it.

    Args:
        name: Tool name to resolve on PATH.
        build_cmd: Builds the argv given the output file path.
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        w: Width of the region in pixels.
        h: Height of the region in pixels.

    Returns:
        A cropped RGB PIL Image of the requested region.

    Raises:
        RuntimeError: If the tool is missing or fails.

    """
    exe = _require_tool(name)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="ocr-tts-capture-")
    os.close(fd)
    os.unlink(path)
    try:
        try:
            proc = subprocess.run(  # noqa: S603
                [exe, *build_cmd(path)],
                capture_output=True,
                timeout=_capture_tool_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{name} timed out") from exc
        if proc.returncode != 0 or not os.path.exists(path):
            stderr = proc.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"{name} failed (rc={proc.returncode}): {stderr}")
        full = Image.open(path)
        full.load()
        return _crop_fullscreen_to_region(full.convert("RGB"), x, y, w, h)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def _capture_region_custom_command(x: int, y: int, w: int, h: int) -> Image.Image:
    """Capture via OCR_TTS_CAPTURE_COMMAND (escape hatch).

    The environment variable may contain ``{x}`` ``{y}`` ``{w}``
    ``{h}`` placeholders; the command must write a PNG to stdout.

    Args:
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        w: Width of the region in pixels.
        h: Height of the region in pixels.

    Returns:
        An RGB PIL Image of the captured screen region.

    Raises:
        RuntimeError: If unset, or the command fails / yields no PNG.

    """
    template = os.environ.get("OCR_TTS_CAPTURE_COMMAND", "").strip()
    if not template:
        raise RuntimeError("OCR_TTS_CAPTURE_COMMAND is not set")
    cmd = shlex.split(template.format(x=x, y=y, w=w, h=h))
    return _run_tool_capture(cmd)


def _pipewire_video_nodes() -> list[str]:
    """Return ids of available PipeWire video source nodes.

    Gamescope exposes its composited Game Mode output as a PipeWire
    node; nodes whose properties mention ``gamescope`` are listed
    first. Requires the ``pw-dump`` utility.

    Returns:
        A possibly empty list of PipeWire node ids (as strings).

    """
    exe = shutil.which("pw-dump")
    if exe is None:
        return []
    try:
        proc = subprocess.run(  # noqa: S603
            [exe],
            capture_output=True,
            timeout=_capture_tool_timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("pw-dump failed: %s", exc)
        return []
    if proc.returncode != 0:
        return []
    try:
        entries = json.loads(proc.stdout.decode(errors="replace"))
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse pw-dump output: %s", exc)
        return []
    plain: list[str] = []
    preferred: list[str] = []
    for entry in entries if isinstance(entries, list) else []:
        info = entry.get("info") or {}
        props = info.get("props") or {}
        media_class = str(props.get("media.class", ""))
        if not media_class.startswith("Video/") or "Sink" in media_class:
            continue
        node_id = str(entry.get("id", ""))
        if not node_id:
            continue
        if "gamescope" in json.dumps(props).lower():
            preferred.append(node_id)
        else:
            plain.append(node_id)
    return preferred + plain


def _capture_region_pipewire(x: int, y: int, w: int, h: int) -> Image.Image:
    """Capture a full screen frame from a PipeWire video source.

    This is the only method that sees actual game content inside a
    gamescope Game Mode session: gamescope publishes its composited
    output on PipeWire and refuses every other readback path
    (no wlr-screencopy, no portal Screenshot, X11 GetImage fails).

    The frame is grabbed with GStreamer's ``pipewiresrc`` and cropped
    to the requested region. The node can be pinned with the
    ``OCR_TTS_PIPEWIRE_NODE`` environment variable; otherwise it is
    auto-detected with ``pw-dump`` (gamescope nodes first).

    Args:
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        w: Width of the region in pixels.
        h: Height of the region in pixels.

    Returns:
        A cropped RGB PIL Image of the requested region.

    Raises:
        RuntimeError: If GStreamer or a video node cannot be found,
            or the pipeline produced no image.

    """
    gst = shutil.which("gst-launch-1.0")
    if gst is None:
        raise RuntimeError("'gst-launch-1.0' executable not found on PATH")
    env_node = os.environ.get("OCR_TTS_PIPEWIRE_NODE", "").strip()
    candidates = [env_node] if env_node else _pipewire_video_nodes()
    if not candidates:
        raise RuntimeError(
            "no PipeWire video node found (is pw-dump installed and "
            "a compositor exposing a capture stream?)"
        )

    fd, path = tempfile.mkstemp(suffix=".png", prefix="ocr-tts-capture-")
    os.close(fd)
    os.unlink(path)
    try:
        last_error = "no attempt made"
        for node in candidates:
            for prop in ("path", "target-object"):
                cmd = [
                    gst,
                    "-q",
                    "pipewiresrc",
                    f"{prop}={node}",
                    "num-buffers=1",
                    "!",
                    "videoconvert",
                    "!",
                    "pngenc",
                    "!",
                    "filesink",
                    f"location={path}",
                ]
                try:
                    proc = subprocess.run(  # noqa: S603
                        cmd,
                        capture_output=True,
                        timeout=_capture_tool_timeout + 15.0,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("gst-launch-1.0 timed out") from exc
                if proc.returncode == 0 and os.path.exists(path):
                    full = Image.open(path)
                    full.load()
                    logger.info("Captured PipeWire node %s via %s", node, prop)
                    return _crop_fullscreen_to_region(full.convert("RGB"), x, y, w, h)
                stderr = proc.stderr.decode(errors="replace").strip()
                last_error = (
                    f"node {node} via {prop} failed (rc={proc.returncode}): {stderr}"
                )
                logger.debug("PipeWire capture %s", last_error)
                with contextlib.suppress(OSError):
                    os.unlink(path)
        raise RuntimeError(f"PipeWire capture failed: {last_error}")
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def _capture_backends(
    x: int, y: int, w: int, h: int
) -> list[tuple[str, Callable[[], Image.Image]]]:
    """Build the ordered list of capture backends to attempt.

    Args:
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        w: Width of the region in pixels.
        h: Height of the region in pixels.

    Returns:
        A list of (backend name, zero-arg capture callable) pairs.

    """
    wayland = _is_wayland()
    backends: list[tuple[str, Callable[[], Image.Image]]] = [
        ("custom command", lambda: _capture_region_custom_command(x, y, w, h)),
    ]
    if wayland and not _portal_unavailable:
        backends.append(("wayland portal", lambda: _capture_region_wayland(x, y, w, h)))
        backends.append(("grim", lambda: _capture_region_grim(x, y, w, h)))
    # PipeWire sees real game content inside gamescope Game Mode,
    # where every other readback path is refused by the compositor.
    backends.append(("pipewire/gst", lambda: _capture_region_pipewire(x, y, w, h)))
    backends.append(("mss/x11", lambda: _capture_region_x11(x, y, w, h)))
    if not wayland:
        backends.append(("maim", lambda: _capture_region_maim(x, y, w, h)))
    backends.extend(
        (
            (
                "scrot",
                lambda: _capture_via_file("scrot", lambda p: ["-o", p], x, y, w, h),
            ),
            (
                "imagemagick import",
                lambda: _capture_via_file(
                    "import", lambda p: ["-window", "root", p], x, y, w, h
                ),
            ),
            (
                "spectacle",
                lambda: _capture_via_file(
                    "spectacle", lambda p: ["-f", "-b", "-n", "-o", p], x, y, w, h
                ),
            ),
            (
                "gnome-screenshot",
                lambda: _capture_via_file(
                    "gnome-screenshot", lambda p: ["-f", p], x, y, w, h
                ),
            ),
        )
    )
    return backends


def capture_region(x: int, y: int, w: int, h: int) -> Image.Image:
    """Capture the specified screen region.

    Backends are tried in order until one returns a non-blank frame:
    the user-provided ``OCR_TTS_CAPTURE_COMMAND`` (if set), the
    xdg-desktop-portal Screenshot API and ``grim`` on Wayland, then
    PipeWire via GStreamer (the only path that captures game content
    inside gamescope Game Mode), then mss/X11 plus external helpers
    such as maim or scrot. This makes capture work on X11, Wayland
    compositors, and nested compositors like gamescope where both
    portal and X11 GetImage fail.

    Args:
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        w: Width of the region in pixels.
        h: Height of the region in pixels.

    Returns:
        An RGB PIL Image of the captured screen region.

    Raises:
        CaptureError: If every backend failed without producing a frame.

    """
    global _portal_unavailable

    errors: list[str] = []
    blank_result: tuple[str, Image.Image] | None = None
    for name, backend in _capture_backends(x, y, w, h):
        try:
            image = backend()
        except Exception as exc:
            logger.warning("%s capture backend failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
            if name == "wayland portal" and "UnknownMethod" in str(exc):
                # Compositor does not implement the Screenshot portal
                # (e.g. gamescope); skip it for subsequent captures.
                _portal_unavailable = True
            continue
        if not image_is_blank(image):
            logger.info("%s capture succeeded (%dx%d)", name, image.width, image.height)
            return image
        if blank_result is None:
            blank_result = (name, image)

    if blank_result is not None:
        name, image = blank_result
        logger.warning("All capture backends returned blank frames; using %s", name)
        return image
    raise CaptureError(errors)


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------


def _run_clip_tool(cmd: list[str], input_bytes: bytes | None = None) -> bool:
    """Run an external clipboard tool, returning success.

    Missing executables are treated as a quiet skip so the next
    backend can be tried; actual failures are logged as warnings.

    Args:
        cmd: The command argv to execute.
        input_bytes: Optional data piped to the command's stdin.

    Returns:
        True when the command exited successfully.

    """
    exe = shutil.which(cmd[0])
    if exe is None:
        logger.debug("clipboard backend '%s' not installed, skipping", cmd[0])
        return False
    try:
        proc = subprocess.run(  # noqa: S603
            [exe, *cmd[1:]],
            input=input_bytes,
            capture_output=True,
            timeout=_capture_tool_timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("clipboard backend '%s' failed: %s", cmd[0], exc)
        return False
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        logger.warning("clipboard backend '%s' failed: %s", cmd[0], stderr)
        return False
    return True


def _text_backends() -> list[list[str]]:
    """Return candidate argv lists for copying plain text."""
    if sys.platform == "darwin":
        return [["pbcopy"]]
    if sys.platform == "win32":
        return [
            ["clip"],
            ["powershell", "-NoProfile", "-Command", "$input | Set-Clipboard"],
        ]
    backends: list[list[str]] = []
    if _is_wayland():
        backends.append(["wl-copy"])
    backends.extend(
        (
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        )
    )
    return backends


def _image_backends(png_path: str | None = None) -> list[list[str]]:
    """Return candidate argv lists for copying a PNG image.

    Args:
        png_path: Path of a temporary PNG file, only needed by the
            macOS and Windows backends which cannot read stdin.

    Returns:
        Backend argv lists in try order.

    """
    if sys.platform == "darwin":
        if png_path is None:
            raise RuntimeError("macOS image clipboard backend requires a PNG file")
        script = (
            f'set the clipboard to (read (POSIX file "{png_path}") as «class PNGf»)'
        )
        return [["osascript", "-e", script]]
    if sys.platform == "win32":
        if png_path is None:
            raise RuntimeError("Windows image clipboard backend requires a PNG file")
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "Add-Type -AssemblyName System.Drawing;"
            f"$img=[System.Drawing.Image]::FromFile('{png_path}');"
            "[System.Windows.Forms.Clipboard]::SetImage($img)"
        )
        return [["powershell", "-NoProfile", "-STA", "-Command", script]]
    backends: list[list[str]] = []
    if _is_wayland():
        backends.append(["wl-copy", "-t", "image/png"])
    backends.append(["xclip", "-selection", "clipboard", "-t", "image/png"])
    return backends


def copy_text(text: str) -> bool:
    """Copy *text* to the system clipboard, best effort.

    Tries the platform-appropriate tools in turn (wl-copy, xclip and
    xsel on Linux, pbcopy on macOS, clip.exe / PowerShell on Windows).

    Args:
        text: The text to place on the clipboard.

    Returns:
        True when some backend reported success, False otherwise.

    """
    payload = text.encode()
    for cmd in _text_backends():
        if _run_clip_tool(cmd, payload):
            logger.info(
                "Copied %d characters to the clipboard via %s", len(text), cmd[0]
            )
            return True
    logger.warning("Could not copy text to the clipboard: no working backend")
    return False


def copy_image(image: Image.Image) -> bool:
    """Copy *image* to the system clipboard as PNG, best effort.

    On macOS and Windows the image is staged through a temporary PNG
    file because those platforms have no stdin-based clipboard tool.

    Args:
        image: The PIL Image to place on the clipboard.

    Returns:
        True when some backend reported success, False otherwise.

    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    png = buffer.getvalue()

    needs_file = sys.platform in {"darwin", "win32"}
    png_path: str | None = None
    if needs_file:
        fd, png_path = tempfile.mkstemp(suffix=".png", prefix="ocr-tts-clip-")
        with os.fdopen(fd, "wb") as handle:
            handle.write(png)

    try:
        for cmd in _image_backends(png_path):
            if _run_clip_tool(cmd, None if needs_file else png):
                logger.info(
                    "Copied %dx%d image to the clipboard", image.width, image.height
                )
                return True
    finally:
        if png_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(png_path)

    logger.warning("Could not copy image to the clipboard: no working backend")
    return False
