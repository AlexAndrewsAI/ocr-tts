"""OCR screen region and immediately speak the extracted text.

Runs the standard OCR region-selection workflow (transparent overlay,
click-drag, capture, Tesseract OCR) and pipes the resulting text
directly to the running TTS server queue.

Typical usage::

    speak-region
    speak-region -v male -s 1.2
    speak-region --lang fra --host localhost --port 9000
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import typer

from ocr_tts import _launch_monotonic
from ocr_tts.ocr_region import (
    OCRConfig,
    capture_selected_region,
    extract_text,
    image_is_blank,
    select_region,
)
from ocr_tts.queue import echo_latency_report, send_speak_request
from ocr_tts.text2speech import DEFAULT_VOICE

logger = logging.getLogger(__name__)

__all__ = ["RegionSpeechResult", "app", "capture_and_queue_region"]


@dataclass
class RegionSpeechResult:
    """Outcome of the region-selection OCR + queue workflow.

    Attributes:
        status: One of ``"ok"``, ``"skipped"``, or ``"error"``.
        reason: Human-readable explanation for non-``ok`` statuses.
        queue_size: Server queue depth after enqueueing (``ok`` only).
        response: Raw server response (present when ``verbose`` is used).
        turnaround: Processing-only turnaround in seconds, excluding the
            user's interactive region selection (verbose mode only).
        breakdown: Ordered ``(label, seconds)`` per-stage timings
            (verbose mode only).

    """

    status: str
    reason: str | None = None
    queue_size: int = 0
    response: dict[str, object] | None = None
    turnaround: float | None = None
    breakdown: list[tuple[str, float]] = field(default_factory=list)


def capture_and_queue_region(
    *,
    voice: str,
    speed: float,
    host: str,
    port: int,
    lang: str,
    tesseract_cmd: str = "tesseract",
    save_image: str | None = None,
    verbose: bool = False,
    script_start: float | None = None,
) -> RegionSpeechResult:
    """Run the interactive region-selection OCR workflow and queue the text.

    A transparent overlay appears; the user click-drags a region which is
    captured, OCR'd with Tesseract, and sent to the running TTS server.

    Args:
        voice: Voice name or alias used by TTS.
        speed: Speech speed multiplier.
        host: TTS server host.
        port: TTS server port.
        lang: OCR language code (e.g., ``eng``, ``fra``).
        tesseract_cmd: Path to the tesseract executable.
        save_image: Optional path to save the captured region image.
        verbose: When True, request latency info from the server and
            compute a processing-only turnaround time.
        script_start: Optional ``time.monotonic()`` start reference for
            the verbose turnaround measurement (callers such as the CLI
            pass the package-import timestamp so launch overhead is
            included); defaults to "now".

    Returns:
        A :class:`RegionSpeechResult` describing the outcome.

    """
    if script_start is None:
        script_start = time.monotonic()
    config = OCRConfig(lang=lang, tesseract_cmd=tesseract_cmd)

    typer.echo("Click and drag to select a screen region...", err=True)
    select_start = time.monotonic()
    ui_ready: float | None = None

    def _on_handoff() -> None:
        nonlocal ui_ready
        ui_ready = time.monotonic()

    region = select_region(on_handoff=_on_handoff)
    select_done = time.monotonic()
    logger.info("Selected region: %s", region)

    # ui_load = loading the region UI up to the handoff to the user.
    # user_select = the user's click-drag interaction (mainloop duration).
    ui_load_s = (ui_ready - select_start) if ui_ready is not None else 0.0
    user_select_s = select_done - (ui_ready if ui_ready is not None else select_start)

    if region.width == 0 or region.height == 0:
        return RegionSpeechResult(status="skipped", reason="no region selected")

    typer.echo(
        f"Capturing region ({region.x}, {region.y}, {region.width}x{region.height})...",
        err=True,
    )
    capture_start = time.monotonic()
    image = capture_selected_region(region)
    capture_s = time.monotonic() - capture_start
    if save_image is not None:
        image.save(save_image)
        typer.echo(f"Saved captured region to {save_image}", err=True)
    if image_is_blank(image):
        typer.echo(
            "Warning: captured region appears blank; screen capture may "
            "have failed on this display server.",
            err=True,
        )
    ocr_start = time.monotonic()
    text = extract_text(image, config=config)
    ocr_s = time.monotonic() - ocr_start

    if not text:
        return RegionSpeechResult(status="error", reason="no text detected")

    typer.echo(f"Extracted text: {text}", err=True)
    response = send_speak_request(
        text,
        host=host,
        port=port,
        voice=voice,
        speed=speed,
        verbose=verbose,
    )
    raw_queue_size = response.get("queue_size", 0)
    queue_size = raw_queue_size if isinstance(raw_queue_size, int) else 0
    typer.echo(
        f"Queued: {queue_size} item(s) pending in the queue",
        err=True,
    )

    result = RegionSpeechResult(status="ok", queue_size=queue_size)
    if verbose:
        total_wall = time.monotonic() - script_start
        # Subtract the user's interactive region selection so the
        # turnaround reflects processing time, not click-drag latency.
        result.turnaround = max(0.0, total_wall - user_select_s)
        result.breakdown = [
            ("region-ui-load", ui_load_s),
            ("user-region-select", user_select_s),
            ("capture", capture_s),
            ("ocr", ocr_s),
        ]
        result.response = response
    return result


app = typer.Typer(
    help="Select a screen region, extract text via OCR, and speak it",
    no_args_is_help=True,
)


@app.command()
def speak_region(
    voice: str = typer.Option(
        DEFAULT_VOICE,
        "--voice",
        "-v",
        help="Voice name or alias (e.g., en_US-hfc_male-medium, male, female)",
    ),
    speed: float = typer.Option(
        1.0,
        "--speed",
        "-s",
        min=0.1,
        max=3.0,
        help="Speech speed multiplier (0.1-3.0, default: 1.0)",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="TTS server host",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        help="TTS server port",
    ),
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help=(
            "Wait for the API to process the item and report latency info "
            "and the total turnaround time."
        ),
    ),
) -> None:
    """Select a screen region, extract text via OCR, and queue it for speech.

    A transparent overlay will appear. Click and drag to select the
    region you want to extract text from. The extracted text is then
    sent to the running TTS server for speech synthesis.

    With ``--verbose`` the command waits for the server to synthesize the
    item and prints a breakdown of the per-stage processing times —
    including the user's interactive region selection (which is *not*
    counted in the turnaround time). The reported ``turnaround-time``
    measures from process launch (captured when the ``ocr_tts`` package is
    first imported, before the heavy TTS imports) until the API returned
    the latency info, **minus** the time the user spent click-dragging to
    select the region (since that is interactive wall-clock, not
    processing).
    """
    script_start = _launch_monotonic
    result = capture_and_queue_region(
        voice=voice,
        speed=speed,
        host=host,
        port=port,
        lang=lang,
        tesseract_cmd=tesseract_cmd,
        save_image=save_image,
        verbose=verbose,
        script_start=script_start,
    )
    if result.status == "skipped":
        typer.echo("No region selected. Exiting.", err=True)
        raise typer.Exit(code=0)
    if result.status == "error":
        typer.echo(f"({result.reason})", err=True)
        raise typer.Exit(code=1)
    if verbose and result.response is not None:
        echo_latency_report(
            result.response,
            script_start,
            turnaround_override=result.turnaround,
            breakdown=result.breakdown,
        )


if __name__ == "__main__":
    app()
