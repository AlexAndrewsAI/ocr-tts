"""Background keystroke watcher for triggering OCR-TTS commands.

Monitors global keyboard events via ``pynput`` and dispatches configured
actions when hotkeys are pressed.  Actions reuse the existing queue client
(``ocr_tts.queue``) so a running TTS server is controlled remotely exactly
as the CLI does.

Typical usage::

    ocr-tts hotkey-watcher start
    ocr-tts hotkey-watcher start --config ~/my-hotkeys.yaml

With no ``--config``, the bundled ``hotkeys.example.yaml`` at the
project root is used.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import BaseModel, Field

from ocr_tts.queue import (
    send_clear_request,
    send_shutdown_request,
    send_speak_request,
)
from ocr_tts.text2speech import DEFAULT_VOICE

__all__ = [
    "app",
    "create_default_config",
    "get_default_config_path",
    "load_config",
    "resolve_config_path",
]

logger = logging.getLogger(__name__)


class HotkeyAction(str, Enum):
    """Supported hotkey actions."""

    SPEAK_TEXT = "speak-text"
    QUEUE_CLEAR = "queue-clear"
    SHUTDOWN = "shutdown"
    OCR_REGION = "ocr-region"
    SEND_REGION = "send-region"
    LAUNCH = "launch"
    SEQUENCE = "sequence"


class HotkeyConfigItem(BaseModel):
    """Configuration for a single hotkey binding."""

    hotkey: str = Field(
        ...,
        description="pynput GlobalHotKeys combination (e.g. '<ctrl>+<shift>+s')",
    )
    action: HotkeyAction = Field(
        ..., description="Action triggered when the hotkey is pressed"
    )
    actions: list[HotkeyAction] = Field(
        default_factory=list,
        description=(
            "Ordered sub-actions for the sequence action "
            "(e.g. [shutdown, launch] to restart the TTS server)"
        ),
    )
    text: str = Field(
        default="Hello from the OCR-TTS hotkey watcher!",
        description="Text spoken by the speak-text action",
    )
    voice: str = Field(
        default=DEFAULT_VOICE,
        description="Voice name or alias used by TTS actions",
    )
    speed: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Speech speed multiplier",
    )
    lang: str = Field(
        default="eng",
        description="OCR language code for the region actions",
    )
    tesseract_cmd: str = Field(
        default="tesseract",
        description="Path to the tesseract executable for the send-region action",
    )
    save_image: str | None = Field(
        default=None,
        description=(
            "Optional path to save the captured region image (send-region action only)"
        ),
    )
    host: str = Field(default="127.0.0.1", description="TTS server host")
    port: int = Field(default=8000, description="TTS server port")


class HotkeyConfig(BaseModel):
    """Complete hotkey watcher configuration."""

    hotkeys: list[HotkeyConfigItem] = Field(
        default_factory=list, description="Configured hotkey bindings"
    )


DEFAULT_CONFIG_FILENAME = "hotkeys.example.yaml"


def get_default_config_path() -> Path:
    """Return the path to the project's bundled example configuration.

    Returns:
        Path to ``hotkeys.example.yaml`` located at the project root
        (resolved relative to this module, never a hardcoded absolute).

    """
    return Path(__file__).resolve().parent.parent / DEFAULT_CONFIG_FILENAME


def resolve_config_path(config_path: str | Path) -> Path:
    """Validate and normalize a configuration file path.

    A leading ``~`` is expanded to the user's home directory via
    :meth:`pathlib.Path.expanduser`, and the resulting path must point
    to an existing file.

    Args:
        config_path: Path to the configuration file (``~`` allowed).

    Returns:
        The expanded, validated path.

    Raises:
        FileNotFoundError: If the path does not exist or is not a file.

    """
    expanded = Path(config_path).expanduser()
    if not expanded.is_file():
        raise FileNotFoundError(f"Configuration file not found: {expanded}")
    return expanded


def load_config(config_path: str | Path) -> HotkeyConfig:
    """Load hotkey configuration from a YAML file.

    The path is ``~``-expanded and validated before parsing; the YAML
    content is validated against the pydantic schema.

    Args:
        config_path: Path to the configuration file (a leading ``~``
            is treated as the user's home directory).

    Returns:
        Parsed :class:`HotkeyConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML content does not match the schema.

    """
    path = resolve_config_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid configuration structure in {path}: expected a mapping"
        )
    return HotkeyConfig(**data)


def create_default_config() -> HotkeyConfig:
    """Create the built-in default hotkey configuration.

    Returns:
        A :class:`HotkeyConfig` with sensible default bindings.

    """
    return HotkeyConfig(
        hotkeys=[
            HotkeyConfigItem(hotkey="<ctrl>+<shift>+s", action=HotkeyAction.SPEAK_TEXT),
            HotkeyConfigItem(
                hotkey="<ctrl>+<shift>+x", action=HotkeyAction.QUEUE_CLEAR
            ),
            HotkeyConfigItem(hotkey="<ctrl>+<shift>+q", action=HotkeyAction.SHUTDOWN),
            HotkeyConfigItem(
                hotkey="<ctrl>+<shift>+o", action=HotkeyAction.SEND_REGION
            ),
        ]
    )


def run_ocr_region(item: HotkeyConfigItem) -> dict[str, Any]:
    """Run the interactive region-selection OCR workflow and queue the text.

    Args:
        item: Hotkey configuration providing voice/speed/lang/server settings.

    Returns:
        Result dictionary describing the outcome.

    """
    from ocr_tts.ocr_region import (
        capture_selected_region,
        extract_text,
        image_is_blank,
        select_region,
    )
    from ocr_tts.text2speech import resolve_voice_alias

    logger.info("Hotkey triggered: ocr-region (%s)", item.hotkey)
    region = select_region()
    if region.width == 0 or region.height == 0:
        return {"status": "skipped", "reason": "no region selected"}
    image = capture_selected_region(region)
    if image_is_blank(image):
        logger.warning("Captured region appears blank")
    text = extract_text(image)
    if not text:
        return {"status": "error", "reason": "no text detected"}
    response = send_speak_request(
        text,
        host=item.host,
        port=item.port,
        voice=resolve_voice_alias(item.voice),
        speed=item.speed,
        verbose=False,
    )
    return {"status": "ok", "queue_size": response.get("queue_size", 0)}


def run_send_region(item: HotkeyConfigItem) -> dict[str, Any]:
    """Run the ``api send-region`` workflow and queue the extracted text.

    Mirrors the ``ocr-tts api send-region`` CLI: interactive region
    selection, capture, Tesseract OCR, then a queue request to the
    running server.  All parameters (voice, speed, lang,
    ``tesseract_cmd``, ``save_image``, host, port) come from the hotkey
    config item.

    Args:
        item: Hotkey configuration providing all send-region settings.

    Returns:
        Result dictionary describing the outcome.

    """
    from ocr_tts.speak_region import capture_and_queue_region

    logger.info("Hotkey triggered: send-region (%s)", item.hotkey)
    result = capture_and_queue_region(
        voice=item.voice,
        speed=item.speed,
        host=item.host,
        port=item.port,
        lang=item.lang,
        tesseract_cmd=item.tesseract_cmd,
        save_image=item.save_image,
        verbose=False,
    )
    if result.status == "ok":
        return {"status": "ok", "queue_size": result.queue_size}
    if result.status == "skipped":
        return {"status": "skipped", "reason": result.reason or "skipped"}
    return {"status": "error", "reason": result.reason or "error"}


def run_launch(item: HotkeyConfigItem) -> dict[str, Any]:
    """Start the TTS API server as a background subprocess.

    Mirrors the ``ocr-tts api launch`` CLI: spawns
    ``python -m ocr_tts.api`` detached with the binding's host/port and
    waits until the server accepts TCP connections.

    Args:
        item: Hotkey configuration providing host and port settings.

    Returns:
        Result dictionary describing the outcome.

    """
    from ocr_tts.queue import _launch_server, _wait_for_server

    logger.info("Hotkey triggered: launch (%s)", item.hotkey)
    proc = _launch_server(item.host, item.port)
    if proc is None:
        return {"status": "error", "reason": "failed to start TTS API server"}
    if _wait_for_server(item.host, item.port):
        return {"status": "ok"}
    return {"status": "error", "reason": "TTS API server did not become ready"}


def execute_action(item: HotkeyConfigItem) -> dict[str, Any]:
    """Execute the action configured for one hotkey binding.

    Args:
        item: The hotkey binding that was activated.

    Returns:
        Result dictionary describing the outcome.  For ``sequence``
        actions the result carries a ``steps`` list with one entry per
        executed sub-action; execution stops at the first step whose
        status is not ``ok``.

    """
    try:
        if item.action is HotkeyAction.SPEAK_TEXT:
            logger.info("Hotkey triggered: speak-text (%s)", item.hotkey)
            response = send_speak_request(
                item.text,
                host=item.host,
                port=item.port,
                voice=item.voice,
                speed=item.speed,
                verbose=False,
            )
            return {"status": "ok", "queue_size": response.get("queue_size", 0)}
        if item.action is HotkeyAction.QUEUE_CLEAR:
            logger.info("Hotkey triggered: queue-clear (%s)", item.hotkey)
            response = send_clear_request(host=item.host, port=item.port)
            return {"status": "ok", "queue_size": response.get("queue_size", 0)}
        if item.action is HotkeyAction.SHUTDOWN:
            logger.info("Hotkey triggered: shutdown (%s)", item.hotkey)
            shutdown_response = send_shutdown_request(host=item.host, port=item.port)
            return {"status": "ok", "response": shutdown_response}
        if item.action is HotkeyAction.OCR_REGION:
            return run_ocr_region(item)
        if item.action is HotkeyAction.SEND_REGION:
            return run_send_region(item)
        if item.action is HotkeyAction.LAUNCH:
            return run_launch(item)
        if item.action is HotkeyAction.SEQUENCE:
            return _execute_sequence(item)
        return {"status": "error", "reason": f"unknown action {item.action!r}"}
    except SystemExit:
        # typer.Exit raised by the queue client on hard failures.
        raise
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.exception("Hotkey action %s failed: %s", item.action, exc)
        return {"status": "error", "reason": str(exc)}


def _execute_sequence(item: HotkeyConfigItem) -> dict[str, Any]:
    """Run a binding's sub-actions in order, stopping at first failure.

    Args:
        item: The hotkey binding carrying the ordered ``actions`` list.

    Returns:
        Result dictionary with an overall status and a per-step ``steps``
        list (each entry keyed by the sub-action name).

    """
    logger.info("Hotkey triggered: sequence %s (%s)", item.actions, item.hotkey)
    steps: list[dict[str, Any]] = []
    for step in item.actions:
        step_item = item.model_copy(update={"action": step})
        result = execute_action(step_item)
        steps.append({"action": step.value, **result})
        if result.get("status") != "ok":
            logger.warning("Sequence stopped at step %r (%s)", step.value, item.hotkey)
            return {"status": "error", "steps": steps}
    return {"status": "ok", "steps": steps}


# Actions that open the tkinter region-selection overlay must all run on
# the *same* worker thread: Tcl/Tk initializes its epoll-based event
# notifier once per process, and creating a second ``Tk()`` from a
# different thread aborts the interpreter ("epoll_ctl: Invalid argument").
# A single-worker executor guarantees every Tk root is created in the
# thread that created the first one.
_UI_ACTION_NAMES = frozenset(
    {HotkeyAction.OCR_REGION.value, HotkeyAction.SEND_REGION.value}
)
_ui_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hotkey-ui")


def dispatch_action(item: HotkeyConfigItem) -> None:
    """Run an action off the pynput listener thread.

    UI-bearing actions (region selection) are submitted to the shared
    single-thread executor; everything else runs on its own daemon
    thread so slow network calls never block each other.

    Args:
        item: The hotkey binding that was activated.

    """
    if item.action.value in _UI_ACTION_NAMES:
        _ui_executor.submit(execute_action, item)
        return
    threading.Thread(
        target=execute_action,
        args=(item,),
        name=f"hotkey-{item.action.value}",
        daemon=True,
    ).start()


def build_callbacks(config: HotkeyConfig) -> dict[str, Callable[[], None]]:
    """Map pynput hotkey strings to zero-argument callback functions.

    Each callback dispatches its action off the pynput listener thread so
    slow actions never block keystroke handling.

    Args:
        config: Parsed hotkey configuration.

    Returns:
        Mapping accepted directly by ``keyboard.GlobalHotKeys``.

    """
    callbacks: dict[str, Callable[[], None]] = {}
    for item in config.hotkeys:

        def _make_runner(bound_item: HotkeyConfigItem) -> Callable[[], None]:
            def runner() -> None:
                dispatch_action(bound_item)

            return runner

        callbacks[item.hotkey] = _make_runner(item)
    return callbacks


class HotkeyWatcher:
    """Wraps ``pynput.keyboard.GlobalHotKeys`` with OCR-TTS action bindings."""

    def __init__(self, config: HotkeyConfig) -> None:
        """Create a watcher for *config*.

        Args:
            config: Hotkey configuration containing action mappings.

        """
        self.config = config
        self._listener: Any | None = None

    @property
    def running(self) -> bool:
        """Whether the underlying listener thread is alive."""
        return self._listener is not None and self._listener.is_alive()

    def start(self) -> None:
        """Start listening for hotkeys and block until stopped."""
        from pynput import keyboard

        callbacks = build_callbacks(self.config)
        logger.info(
            "Starting hotkey watcher with %d binding(s): %s",
            len(callbacks),
            ", ".join(callbacks),
        )
        listener = keyboard.GlobalHotKeys(callbacks)
        self._listener = listener
        listener.start()
        try:
            listener.join()
        except KeyboardInterrupt:
            logger.info("Hotkey watcher interrupted by user")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the listener, if running."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            logger.info("Hotkey watcher stopped")


app = typer.Typer(
    help="Manage the background keystroke watcher for OCR-TTS automation",
    no_args_is_help=True,
)


@app.callback()
def _callback() -> None:
    """Manage the background keystroke watcher for OCR-TTS automation."""


@app.command("start")
def start(
    config_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help=(
            "Path to a YAML configuration file (a leading ~ is expanded "
            "to your home directory; defaults to the bundled "
            "hotkeys.example.yaml)"
        ),
    ),
) -> None:
    """Start watching global hotkeys and dispatch configured actions."""
    try:
        if config_path is not None:
            config = load_config(config_path)
        else:
            config = load_config(get_default_config_path())
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1) from None

    watcher = HotkeyWatcher(config)
    typer.echo("Hotkey watcher running. Press Ctrl+C to stop.", err=True)
    try:
        watcher.start()
    except Exception as exc:
        typer.echo(f"Failed to start hotkey watcher: {exc}", err=True)
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
