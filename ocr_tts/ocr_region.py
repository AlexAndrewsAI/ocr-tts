"""OCR region selection and text extraction module.

Provides functionality to select a screen region via mouse
drag, capture the region, and extract text using Tesseract OCR.

OS-dependent services live in :mod:`ocr_tts.desktop`:

* screen capture through a chain of backends covering X11,
  Wayland compositors and nested compositors such as gamescope:
  ``OCR_TTS_CAPTURE_COMMAND`` -> xdg-desktop-portal -> ``grim``
  -> PipeWire/GStreamer (the only path that captures real game
  content inside gamescope Game Mode) -> mss/X11 ->
  ``maim``/``scrot``/``import``/``spectacle``/``gnome-screenshot``.
* clipboard access (:func:`copy_text`, :func:`copy_image`).

Captured frames may be at a different pixel resolution than the
logical screen (e.g. gamescope PipeWire output vs XWayland logical
size); region coordinates are mapped between the two spaces where
needed.

The selection overlay shows a frozen screenshot when one can be
captured; otherwise it falls back to a translucent window so the
user can still aim at the live screen content underneath.

OCR always crops the frozen screenshot the user selected on when it
is usable, because it is pixel-exact with respect to what was shown;
a fresh grab is only taken when no frozen frame exists, and can then
drift from the selection under compositors whose capture resolution
differs from the logical screen (e.g. gamescope Game Mode).
"""

import logging
import time
from collections.abc import Callable
from typing import Any, NamedTuple

import pytesseract
import typer
from PIL import Image
from pydantic import BaseModel, Field

from ocr_tts.desktop import (
    _primary_monitor,
    capture_region,
    image_is_blank,
)

logger = logging.getLogger(__name__)

_selection_background: Image.Image | None = None
_selection_monitor: dict[str, int] | None = None


class Region(NamedTuple):
    """Represents a screen region for capture.

    Attributes:
        x: Left coordinate of the region.
        y: Top coordinate of the region.
        width: Width of the region in pixels.
        height: Height of the region in pixels.

    """

    x: int
    y: int
    width: int
    height: int


class OCRConfig(BaseModel):
    """Configuration for OCR operations.

    Attributes:
        tesseract_cmd: Path to the tesseract executable.
        lang: OCR language code (e.g., 'eng', 'fra', 'deu').

    """

    tesseract_cmd: str = Field(
        default="tesseract", description="Path to the tesseract executable"
    )
    lang: str = Field(default="eng", description="OCR language code")

    model_config = {"title": "OCR Config", "frozen": True}


def _capture_background(monitor: dict[str, int]) -> Image.Image | None:
    """Capture the screen behind the selection overlay.

    Args:
        monitor: mss-style monitor bounds dict.

    Returns:
        A PIL Image of the screen, or None if capture fails.

    """
    try:
        for attempt in range(2):
            try:
                image = capture_region(
                    monitor["left"],
                    monitor["top"],
                    monitor["width"],
                    monitor["height"],
                )
            except Exception as exc:
                logger.warning(
                    "Could not capture screen for selection overlay: %s", exc
                )
                time.sleep(0.3)
                continue
            if not image_is_blank(image):
                return image
            logger.warning(
                "Screen capture returned a blank frame on attempt %d", attempt + 1
            )
            time.sleep(0.3)
    except Exception as exc:
        logger.warning("Could not capture screen for selection overlay: %s", exc)
        return None
    return None


class _SelectionApp:
    """Fullscreen region-selection overlay implemented with tkinter.

    Renders a borderless always-on-top window showing the captured
    screen (or a translucent window when capture fails) and tracks
    mouse press, drag and release to build a Region.
    """

    def __init__(
        self,
        tkmod: Any,
        monitor: dict[str, int],
        background: Image.Image | None,
    ) -> None:
        self._tk = tkmod
        self._monitor = monitor
        self._background = background
        self._root: Any = tkmod.Tk()
        self._canvas: Any = None
        self._photo: Any = None
        self._rect: Any = None
        self._start_x: int | None = None
        self._start_y: int | None = None
        self.region: Region | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        """Create and configure the fullscreen overlay window."""
        left = int(self._monitor["left"])
        top = int(self._monitor["top"])
        width = int(self._monitor["width"])
        height = int(self._monitor["height"])

        root = self._root
        root.geometry(f"{width}x{height}+{left}+{top}")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()

        canvas = self._tk.Canvas(
            root,
            width=width,
            height=height,
            cursor="crosshair",
            highlightthickness=0,
        )
        canvas.pack()
        if self._background is not None:
            from PIL import ImageTk

            # The captured frame may be at a different resolution than
            # the logical screen (e.g. gamescope PipeWire output vs
            # XWayland logical size), so stretch it to exactly cover
            # the overlay window instead of clipping it.
            display = self._background
            if display.size != (width, height):
                display = display.resize((width, height))
            self._photo = ImageTk.PhotoImage(display)
            canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            # No frozen screenshot available (e.g. gamescope or a
            # locked-down Wayland compositor). Use a translucent
            # window so the live screen stays visible behind the
            # selection rectangle.
            root.attributes("-alpha", 0.30)
            canvas.config(bg="#101010")
            canvas.create_text(
                width // 2,
                height // 2,
                text=(
                    "Click and drag to select a screen region.\nPress Escape to cancel."
                ),
                fill="white",
                font=("DejaVu Sans", 22),
                justify="center",
            )
        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)
        canvas.bind("<Escape>", self._on_escape)
        root.bind("<Escape>", self._on_escape)
        self._canvas = canvas

    def _screen_coords(self, event: Any) -> tuple[int, int]:
        """Convert a canvas event to absolute screen coordinates."""
        x = int(self._monitor["left"]) + int(event.x)
        y = int(self._monitor["top"]) + int(event.y)
        return x, y

    def _on_press(self, event: Any) -> None:
        """Record the drag start and draw the selection rectangle."""
        self._start_x, self._start_y = self._screen_coords(event)
        self._rect = self._canvas.create_rectangle(
            self._start_x,
            self._start_y,
            self._start_x,
            self._start_y,
            outline="red",
            width=2,
            dash=(4, 2),
        )

    def _on_drag(self, event: Any) -> None:
        """Update the selection rectangle while dragging."""
        if (
            self._rect is not None
            and self._start_x is not None
            and self._start_y is not None
        ):
            x, y = self._screen_coords(event)
            self._canvas.coords(self._rect, self._start_x, self._start_y, x, y)

    def _on_release(self, event: Any) -> None:
        """Finish the selection and close the overlay."""
        if self._start_x is None or self._start_y is None:
            self._root.quit()
            return
        x, y = self._screen_coords(event)
        self.region = Region(
            x=min(self._start_x, x),
            y=min(self._start_y, y),
            width=abs(x - self._start_x),
            height=abs(y - self._start_y),
        )
        self._root.quit()

    def _on_escape(self, _event: Any) -> None:
        """Cancel the selection with an empty region."""
        self.region = Region(x=0, y=0, width=0, height=0)
        self._root.quit()

    def run(self) -> None:
        """Run the tkinter main loop and tear down the overlay.

        The window is withdrawn and the compositor given time to
        repaint the underlying screen before it is destroyed, so a
        subsequent screen capture returns the live screen rather than
        the stale overlay frame.
        """
        self._root.mainloop()
        self._root.withdraw()
        self._root.update_idletasks()
        self._root.update()
        time.sleep(0.3)
        self._root.destroy()


def select_region(on_handoff: Callable[[], None] | None = None) -> Region:
    """Display a fullscreen transparent overlay for region selection.

    The user clicks and drags to select a rectangular region
    on the screen. The overlay is rendered with tkinter, which
    delivers its own mouse events and therefore works on both
    X11 and Wayland (via XWayland) compositors.

    Args:
        on_handoff: Optional no-argument callback invoked immediately
            before the tkinter main loop starts — i.e. the precise moment
            the selection UI is handed off to the user. Used by callers
            that want to measure UI load time separately from the user's
            click-drag selection time.

    Returns:
        A Region namedtuple with (x, y, width, height). If the
        selection is cancelled or the overlay cannot be created,
        an empty Region is returned.

    """
    try:
        import tkinter as tk
    except ImportError as exc:
        logger.warning("tkinter is required for region selection: %s", exc)
        return Region(x=0, y=0, width=0, height=0)

    global _selection_background, _selection_monitor
    _selection_background = None
    _selection_monitor = None

    try:
        monitor = _primary_monitor()
    except Exception as exc:
        logger.warning("Could not determine monitor bounds: %s", exc)
        return Region(x=0, y=0, width=0, height=0)

    background = _capture_background(monitor)
    _selection_background = background
    _selection_monitor = monitor

    try:
        app = _SelectionApp(tk, monitor, background)
    except tk.TclError as exc:
        logger.warning("Could not open a display for region selection: %s", exc)
        return Region(x=0, y=0, width=0, height=0)

    if on_handoff is not None:
        on_handoff()
    app.run()
    if app.region is None:
        return Region(x=0, y=0, width=0, height=0)
    return app.region


def _crop_selection_background(region: Region) -> Image.Image | None:
    """Crop the pre-capture background frame to the selected region.

    The background frame is the screen capture shown to the user
    inside the selection overlay. If it is available it provides an
    exact picture of what the overlay covered, so it can rescue a
    selection whose live capture returned a blank frame.

    The frame may be at a different pixel resolution than the logical
    screen (gamescope output vs XWayland logical size), so the
    logical screen coordinates are mapped into background pixels.

    Args:
        region: The selected screen region.

    Returns:
        The cropped background image, or None when unavailable.

    """
    if _selection_background is None or _selection_monitor is None:
        return None
    monitor_w = max(1, int(_selection_monitor["width"]))
    monitor_h = max(1, int(_selection_monitor["height"]))
    scale_x = _selection_background.width / monitor_w
    scale_y = _selection_background.height / monitor_h
    left = round((region.x - int(_selection_monitor["left"])) * scale_x)
    top = round((region.y - int(_selection_monitor["top"])) * scale_y)
    width_px = round(region.width * scale_x)
    height_px = round(region.height * scale_y)
    if (
        left < 0
        or top < 0
        or left + width_px > _selection_background.width
        or top + height_px > _selection_background.height
    ):
        return None
    return _selection_background.crop((left, top, left + width_px, top + height_px))


def capture_selected_region(region: Region) -> Image.Image:
    """Capture a selected region, preferring the frozen overlay frame.

    The pre-capture frame shown inside the selection overlay is the
    exact picture the user aimed at, and cropping it is a pixel-exact
    inverse of the stretch used to display it. A fresh grab after the
    overlay closes can instead drift from the selection when the live
    frame's resolution or extent differs from the logical screen the
    coordinates were captured in -- for example under gamescope Game
    Mode, where the PipeWire stream may not match the XWayland logical
    size, or simply because the game content moved on. The frozen
    frame is therefore returned whenever a usable crop exists; the
    live capture chain is only consulted as a fallback.

    Args:
        region: The region selected by the user.

    Returns:
        An RGB PIL Image of the selected region.

    Raises:
        CaptureError: If no usable overlay frame exists and every
            capture backend failed.

    """
    frozen = _crop_selection_background(region)
    if frozen is not None and not image_is_blank(frozen):
        logger.info(
            "Using the %dx%d frame shown during selection (frozen-frame policy)",
            frozen.width,
            frozen.height,
        )
        return frozen

    if frozen is None:
        logger.info("No frozen overlay frame available; taking a fresh screen grab")
    else:
        logger.info("Frozen overlay frame was blank; taking a fresh screen grab")

    image = capture_region(region.x, region.y, region.width, region.height)
    if not image_is_blank(image):
        return image
    logger.warning(
        "Fresh screen capture was blank; reusing the frame shown during selection"
    )
    fallback = _crop_selection_background(region)
    if fallback is not None and not image_is_blank(fallback):
        return fallback
    return image


def extract_text(image: Image.Image, config: OCRConfig | None = None) -> str:
    """Extract text from an image using Tesseract OCR.

    Args:
        image: The PIL Image to process.
        config: Optional OCR configuration. If not provided,
                the default config is used.

    Returns:
        The extracted text string, stripped of leading/trailing whitespace.

    """
    if config is None:
        config = OCRConfig()

    pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
    raw_text: object = pytesseract.image_to_string(image, lang=config.lang)
    text = str(raw_text)
    logger.info("Extracted %d characters via OCR", len(text))
    return text.strip()


def ocr_region_command(
    lang: str = typer.Option(
        "eng",
        "--lang",
        "-l",
        help="OCR language code (e.g., eng, fra, deu, spa)",
    ),
    tesseract_cmd: str = typer.Option(
        "tesseract",
        "--tesseract-cmd",
        help="Path to the tesseract executable",
    ),
    save_image: str | None = typer.Option(
        None,
        "--save-image",
        help="Save the captured region to this image file for inspection",
    ),
) -> None:
    """Select a screen region and extract text via OCR.

    A transparent overlay will appear. Click and drag to
    select the region you want to extract text from.
    """
    config = OCRConfig(lang=lang, tesseract_cmd=tesseract_cmd)

    typer.echo("Click and drag to select a screen region...", err=True)
    region = select_region()
    logger.info("Selected region: %s", region)

    if region.width == 0 or region.height == 0:
        typer.echo("No region selected. Exiting.", err=True)
        raise typer.Exit(code=0)

    typer.echo(
        f"Capturing region ({region.x}, {region.y}, {region.width}x{region.height})...",
        err=True,
    )
    image = capture_selected_region(region)
    if save_image is not None:
        image.save(save_image)
        typer.echo(f"Saved captured region to {save_image}", err=True)
    if image_is_blank(image):
        typer.echo(
            "Warning: captured region appears blank; screen capture may "
            "have failed on this display server.",
            err=True,
        )
    text = extract_text(image, config=config)

    if text:
        typer.echo(text)
    else:
        typer.echo("(no text detected)", err=True)
