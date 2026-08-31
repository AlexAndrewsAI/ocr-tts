"""Command line interface for OCR-TTS.

Consolidates every OCR-TTS entry point into a single ``ocr-tts``
Typer CLI.  All subcommands are available via ``uv run ocr-tts ...``:

* ``ocr`` -- select a screen region and extract text via OCR.
* ``text2speech`` -- convert text to a WAV file via Piper TTS.
* ``api`` -- manage the running TTS server queue:

  - ``launch`` -- run the FastAPI server.
  - ``send-text`` -- add text to the running server queue.
  - ``clear`` -- wipe the queue and stop playback immediately.
  - ``send-region`` -- select a screen region, OCR it, and queue the text.
  - ``close`` -- tear down the running server and its subprocesses.

* ``live`` -- stream text (from stdin) to TTS and play it back immediately.
* ``hotkey-watcher`` -- run a background global-hotkey service:

  - ``start`` -- watch configured hotkeys and dispatch OCR-TTS actions
    (defaults to the bundled ``hotkeys.example.yaml``).

The command bodies are thin re-registrations / wrappers around the
existing per-module entry points, so behaviour is unchanged.
"""

import logging
from importlib.metadata import version

import typer

from ocr_tts.api import serve as _api_serve
from ocr_tts.desktop import copy_image, copy_text
from ocr_tts.hotkey_watcher import app as _hotkey_watcher_app
from ocr_tts.ocr_region import (
    OCRConfig,
    capture_selected_region,
    extract_text,
    image_is_blank,
    select_region,
)
from ocr_tts.player import live as _live
from ocr_tts.queue import clear as _clear
from ocr_tts.queue import close as _close
from ocr_tts.queue import speak as _send_text
from ocr_tts.speak_region import speak_region as _send_region
from ocr_tts.text2speech import main as _text2speech

__version__ = version("ocr-tts")

logger = logging.getLogger(__name__)

__all__ = ["app"]

app = typer.Typer(help="OCR and TTS toolkit", no_args_is_help=True)


def version_callback(value: bool) -> None:
    """Handle the version flag callback."""
    if value:
        typer.echo(f"ocr-tts {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """OCR and TTS toolkit.

    Run ``ocr-tts --help`` to list the available subcommands.
    """
    del version  # handled eagerly by version_callback


@app.command("ocr")
def ocr(
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
    copy_text_flag: bool = typer.Option(
        False,
        "--copy-text",
        help="Copy the extracted text to the system clipboard",
    ),
    copy_image_flag: bool = typer.Option(
        False,
        "--copy-image",
        help="Copy the captured region image to the system clipboard",
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
    if copy_image_flag:
        if copy_image(image):
            typer.echo("Captured region copied to the clipboard", err=True)
        else:
            typer.echo(
                "Could not copy the captured region to the clipboard",
                err=True,
            )
    if image_is_blank(image):
        typer.echo(
            "Warning: captured region appears blank; screen capture may "
            "have failed on this display server.",
            err=True,
        )
    text = extract_text(image, config)

    if text:
        if copy_text_flag:
            if copy_text(text):
                typer.echo("Extracted text copied to the clipboard", err=True)
            else:
                typer.echo(
                    "Could not copy the extracted text to the clipboard",
                    err=True,
                )
        typer.echo(text)
    else:
        typer.echo("(no text detected)", err=True)


# text2speech and live are re-registered from their feature modules so
# the command signatures and behaviour stay identical.
app.command("text2speech")(_text2speech)
app.command("live")(_live)

# The api sub-command group nests the server launcher and the remote
# queue-control commands under ``ocr-tts api ...``.
api_app = typer.Typer(
    help="Manage the running TTS server queue",
    no_args_is_help=True,
)
app.add_typer(api_app, name="api")


@api_app.command("launch")
def launch(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address"),  # noqa: S104
    port: int = typer.Option(8000, "--port", help="Listen port"),
) -> None:
    """Run the API server with uvicorn."""
    _api_serve(host=host, port=port)


api_app.command("send-text")(_send_text)
api_app.command("send-region")(_send_region)
api_app.command("clear")(_clear)
api_app.command("close")(_close)

# The hotkey-watcher group runs the background keystroke service.
app.add_typer(_hotkey_watcher_app, name="hotkey-watcher")


if __name__ == "__main__":
    app()
