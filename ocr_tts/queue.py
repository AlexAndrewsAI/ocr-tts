"""Remote control for the running TTS server queue.

Provides a Typer client for the ``tts-api`` server: ``speak`` appends
text to the server's processing queue (each item carrying its own voice
and speed), and ``clear`` wipes the queue and immediately stops playback.

The queue lives on the server, so the client is intentionally stateless:
it simply forwards the requested text/voice/speed to the running server
and reports the server's response.

Typical usage::

    speak "Hello world"
    speak "Bonjour!" -v fr_FR-siwis-medium -s 1.2
    clear
    close

When the server is not running and a connection is refused, ``speak``
and ``clear`` log a warning, launch ``tts-api`` in the background, and
retry the request once.  A second connection-refused error is treated as
a hard failure.  ``close`` asks the running server to tear itself down
via its ``/shutdown`` endpoint (and force-terminates it as a safety net
if it does not exit).
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import typer

from ocr_tts import __version__, _launch_monotonic
from ocr_tts.text2speech import DEFAULT_VOICE, resolve_voice_alias

logger = logging.getLogger(__name__)

__all__ = [
    "app",
    "clear",
    "close",
    "echo_latency_report",
    "send_clear_request",
    "send_shutdown_request",
    "send_speak_request",
]

app = typer.Typer(
    help="Control the running TTS server queue",
    no_args_is_help=True,
)

_QUEUE_PATH = "queue"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000


def _server_url(host: str, port: int, path: str) -> str:
    """Build a server URL for the given host, port, and path.

    Args:
        host: Server host.
        port: Server port.
        path: Endpoint path (no leading slash).

    Returns:
        The full ``http://host:port/path`` URL.

    """
    return f"http://{host}:{port}/{path}"


def _is_connection_refused(exc: urllib.error.URLError) -> bool:
    """Return True when *exc* represents a connection-refused error.

    Args:
        exc: The URL error raised by ``urllib``.

    Returns:
        True if the underlying reason is ``ConnectionRefusedError`` or an
        ``OSError`` whose errno corresponds to a refused/unreachable
        connection.

    """
    reason = exc.reason
    if isinstance(reason, ConnectionRefusedError):
        return True
    if isinstance(reason, OSError):
        return reason.errno in (
            errno.ECONNREFUSED,
            errno.ENETUNREACH,
            errno.EHOSTUNREACH,
        )
    # Fallback: inspect string representation (e.g. "[Errno 111] Connection refused")
    return "connection refused" in str(reason).lower()


def _parse_host_port(url: str) -> tuple[str, int]:
    """Extract the host and port from *url*.

    Args:
        url: Full endpoint URL.

    Returns:
        A ``(host, port)`` tuple, falling back to the module defaults
        when either component is absent.

    """
    parsed = urlparse(url)
    return parsed.hostname or _DEFAULT_HOST, parsed.port or _DEFAULT_PORT


def _wait_for_server(host: str, port: int, timeout: float = 10.0) -> bool:
    """Poll until a TCP connection to the server succeeds or *timeout* elapses.

    Args:
        host: Server hostname or IP.
        port: Server port.
        timeout: Maximum seconds to wait.

    Returns:
        True if the server accepted a connection, False otherwise.

    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _launch_server(host: str, port: int) -> subprocess.Popen[bytes] | None:
    """Start ``tts-api`` as a background subprocess.

    Args:
        host: Bind address for the new server.
        port: Listen port for the new server.

    Returns:
        The ``Popen`` handle if the process was spawned, or ``None`` on
        failure.

    """
    try:
        return subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "ocr_tts.api",
                "--host",
                host,
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        logger.exception("Failed to launch TTS API server")
        return None


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    """POST a JSON payload to a URL and return the parsed response.

    If the server cannot be reached because the connection is refused,
    the function logs a warning, launches ``tts-api`` in the background,
    waits for it to become ready, and retries the request exactly once.
    A second connection-refused error raises ``typer.Exit`` with code 1.

    Args:
        url: Full endpoint URL.
        payload: JSON-serializable request body.

    Returns:
        The decoded JSON response body.

    Raises:
        typer.Exit: If the server cannot be reached or returns an error.

    """
    request = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
            # Break out of the retry loop on success so the request is
            # sent exactly once.  Previously the loop fell through and
            # resubmitted the same POST on the next iteration, causing
            # every message to be sent (and spoken) twice.
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            typer.echo(f"Server error ({exc.code}): {body}", err=True)
            raise typer.Exit(code=1) from None
        except urllib.error.URLError as exc:
            if attempt == 0 and _is_connection_refused(exc):
                host, port = _parse_host_port(url)
                logger.warning(
                    "Cannot reach TTS server at %s: %s. Attempting to launch server...",
                    url,
                    exc.reason,
                )
                typer.echo(
                    f"Cannot reach TTS server at {url}: {exc.reason}. "
                    "Attempting to launch tts-api in the background...",
                    err=True,
                )
                proc = _launch_server(host, port)
                if proc is not None and _wait_for_server(host, port):
                    logger.info("TTS API server launched, retrying request")
                    continue
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                typer.echo(
                    "Failed to launch TTS API server. Please start it manually.",
                    err=True,
                )
                raise typer.Exit(code=1) from None
            typer.echo(f"Cannot reach TTS server at {url}: {exc.reason}", err=True)
            raise typer.Exit(code=1) from None
    return cast(dict[str, object], data)


def send_speak_request(
    text: str,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
    verbose: bool = False,
) -> dict[str, object]:
    """Enqueue text (with its own voice and speed) on the server.

    Args:
        text: Text to speak.
        host: TTS server host.
        port: TTS server port.
        voice: Piper voice for this item.
        speed: Speed multiplier for this item.
        verbose: When True, ask the server to wait until the item has been
            synthesized and return latency info in the response.

    Returns:
        The server's JSON response.

    """
    resolved_voice = resolve_voice_alias(voice)
    url = _server_url(host, port, _QUEUE_PATH)
    logger.info(
        "Sending speak request to %s: text=%r, voice=%s, speed=%.1f",
        url,
        text[:80],
        resolved_voice,
        speed,
    )
    payload: dict[str, object] = {
        "text": text,
        "voice": resolved_voice,
        "speed": speed,
    }
    if verbose:
        payload["wait"] = True
    response = _post_json(url, payload)
    logger.info("Server response: %s", response)
    return response


def send_clear_request(
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
) -> dict[str, object]:
    """Ask the server to wipe the queue and stop playback immediately.

    Args:
        host: TTS server host.
        port: TTS server port.

    Returns:
        The server's JSON response.

    """
    return _post_json(_server_url(host, port, f"{_QUEUE_PATH}/clear"), {})


def send_shutdown_request(
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
) -> dict[str, object] | None:
    """Ask the running server to tear itself down and exit.

    POSTs to the server's ``/shutdown`` endpoint; a running server stops
    its queue processor and playback thread, drains pending work, and
    exits the uvicorn process gracefully (including any subprocesses it
    started).  Unlike :func:`_post_json`, this never auto-launches a new
    server when the connection is refused.

    Args:
        host: TTS server host.
        port: TTS server port.

    Returns:
        The server's JSON response, or ``None`` when no server is running
        (the connection was refused).

    Raises:
        typer.Exit: If the server returns an error or cannot otherwise be
            reached.

    """
    url = _server_url(host, port, "shutdown")
    request = urllib.request.Request(  # noqa: S310
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return cast(dict[str, object], json.loads(response.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        typer.echo(f"Server error ({exc.code}): {body}", err=True)
        raise typer.Exit(code=1) from None
    except urllib.error.URLError as exc:
        if _is_connection_refused(exc):
            logger.info("TTS server not running; nothing to shut down")
            return None
        typer.echo(f"Cannot reach TTS server at {url}: {exc.reason}", err=True)
        raise typer.Exit(code=1) from None


# How long to wait for a server process to shut down gracefully before
# force-killing it.
_CLOSE_TIMEOUT_S = 5.0


def _find_api_pids(port: int) -> list[int]:
    """Locate the PIDs of running TTS API server processes on ``port``.

    The queue client launches the server as a detached background
    subprocess (``python -m ocr_tts.api --port <port>``), so running
    servers are located by scanning ``/proc`` for a command line that
    runs the ``ocr_tts.api`` module on the requested port.

    Args:
        port: The server port to match.

    Returns:
        A sorted list of PIDs, empty if none are running.

    """
    pids: list[int] = []
    try:
        proc_dirs = list(Path("/proc").iterdir())
    except OSError:
        return pids
    port_token = str(port).encode()
    for proc_dir in proc_dirs:
        if not proc_dir.name.isdigit():
            continue
        try:
            cmdline = (proc_dir / "cmdline").read_bytes()
        except OSError:
            continue
        if b"ocr_tts.api" in cmdline and port_token in cmdline:
            pids.append(int(proc_dir.name))
    return sorted(pids)


def _pid_exists(pid: int) -> bool:
    """Return True if a process with ``pid`` is still running.

    Args:
        pid: The process ID to check.

    Returns:
        True if the process exists, False otherwise.

    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_pid(pid: int, sig: int) -> None:
    """Send ``sig`` to ``pid`` (and its group if it is the group leader).

    The background server is started with ``start_new_session=True``, so
    it is a process-group leader; when that holds, signalling the group
    also tears down any subprocesses the server spawned.  Signalling only
    the leader's group when it is *not* a group leader could hit an
    unrelated group, so an individual kill is used in that case.

    Args:
        pid: The process ID to signal.
        sig: The signal number to send.

    """
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _reap_pid(pid: int) -> None:
    """Reap a reaped-for child process to avoid zombie accumulation.

    Args:
        pid: The process ID to reap.

    """
    with contextlib.suppress(ChildProcessError, OSError):
        os.waitpid(pid, os.WNOHANG)


def _terminate_pid(pid: int) -> None:
    """Gracefully terminate a server process and its subprocess group.

    Sends ``SIGTERM`` (which uvicorn handles cleanly), waits up to
    ``_CLOSE_TIMEOUT_S`` for the process to exit, then force-kills it
    with ``SIGKILL`` if it is still running.

    Args:
        pid: The process ID of the server to shut down.

    """
    deadline = time.monotonic() + _CLOSE_TIMEOUT_S
    _signal_pid(pid, signal.SIGTERM)
    while _pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _pid_exists(pid):
        _signal_pid(pid, signal.SIGKILL)
    _reap_pid(pid)


def _wait_for_pids_gone(pids: list[int], timeout: float = _CLOSE_TIMEOUT_S) -> None:
    """Wait until none of ``pids`` are running, or ``timeout`` elapses.

    Args:
        pids: Process IDs to wait on.
        timeout: Maximum seconds to wait.

    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(_pid_exists(p) for p in pids):
        time.sleep(0.05)


def close(
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
) -> None:
    """Tear down the running TTS API server and its subprocesses.

    Sends a ``POST /shutdown`` command to the running server so it can tear
    itself down internally: it stops the queue processor and playback
    thread, drains pending work, and exits the uvicorn process (including
    any subprocesses it started).  If the server is not running, this is a
    no-op.  As a safety net, any server process that does not exit after
    the command is sent is force-terminated.

    Args:
        host: TTS server host.
        port: TTS server port.

    Raises:
        typer.Exit: If the server returns an error, or its processes
            cannot be signalled because they belong to a different user.

    """
    response = send_shutdown_request(host=host, port=port)
    if response is None:
        typer.echo(
            f"No running TTS API server found on port {port} "
            f"(host {host}). Nothing to close.",
            err=True,
        )
        return

    typer.echo(
        f"Sent shutdown command; waiting for TTS API server on port {port} to exit...",
        err=True,
    )
    # Give the server a chance to tear itself down gracefully, then
    # force-terminate any process that did not exit on its own.
    pids = _find_api_pids(port)
    _wait_for_pids_gone(pids)
    stragglers = [pid for pid in pids if _pid_exists(pid)]
    for pid in stragglers:
        logger.warning(
            "Server did not exit after shutdown command; terminating PID %d", pid
        )
        try:
            _terminate_pid(pid)
        except PermissionError as exc:
            typer.echo(f"Permission denied closing process {pid}: {exc}", err=True)
            raise typer.Exit(code=1) from None
    typer.echo("TTS API server closed.", err=True)


def clear(
    host: str = typer.Option(
        _DEFAULT_HOST,
        "--host",
        help="TTS server host",
    ),
    port: int = typer.Option(
        _DEFAULT_PORT,
        "--port",
        help="TTS server port",
    ),
) -> None:
    """Wipe the queue and immediately stop playback.

    Sends a ``POST /queue/clear`` request to the running server, which
    drains all pending text and audio and stops playback of any in-flight
    item.  Like ``speak``, this auto-launches the server in the background
    when one is not already running.

    Args:
        host: TTS server host.
        port: TTS server port.

    """
    response = send_clear_request(host=host, port=port)
    typer.echo(
        f"Queue cleared; {response.get('queue_size', 0)} item(s) pending",
        err=True,
    )


def _version_callback(value: bool) -> None:
    """Handle the version flag callback."""
    if value:
        typer.echo(f"speak {__version__}")
        raise typer.Exit()


@app.command()
def speak(
    text: str | None = typer.Argument(
        None,
        help="Text to add to the running queue",
    ),
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
        help="Speech speed multiplier (0.5-2.0)",
    ),
    host: str = typer.Option(
        _DEFAULT_HOST,
        "--host",
        help="TTS server host",
    ),
    port: int = typer.Option(
        _DEFAULT_PORT,
        "--port",
        help="TTS server port",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help=(
            "Wait for the API to process the item and report latency info "
            "and the total turnaround time."
        ),
    ),
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Add text to the running TTS server queue.

    Text is appended after any already-queued text.  Each item carries
    its own ``--voice`` and ``--speed``, so switching voice/speed
    mid-queue affects only text submitted after the switch; text already
    in the queue keeps the settings it was submitted with.

    To wipe the queue and stop playback immediately, use the separate
    ``clear`` command (``ocr-tts api clear``).

    With ``--verbose`` the command waits for the server to synthesize the
    item, prints the latency the API reports, and prints the total
    turnaround time measured from process launch (captured when the
    ``ocr_tts`` package is first imported, before the heavy TTS imports)
    until the API returned the latency info.

    Examples:
        speak "Hello, world!"
        speak "Bonjour!" -v fr_FR-siwis-medium -s 1.2
        speak --verbose "Hello, world!"

    """
    script_start = _launch_monotonic
    if version:
        return
    if not text or not text.strip():
        typer.echo("Error: TEXT is required", err=True)
        raise typer.Exit(code=1)
    response = send_speak_request(
        text,
        host=host,
        port=port,
        voice=voice,
        speed=speed,
        verbose=verbose,
    )
    typer.echo(
        f"Queued: {response.get('queue_size', 0)} item(s) pending in the queue",
        err=True,
    )
    if verbose:
        echo_latency_report(response, script_start)


def echo_latency_report(
    response: dict[str, object],
    script_start: float,
    turnaround_override: float | None = None,
    breakdown: list[tuple[str, float]] | None = None,
) -> None:
    """Print the API latency info and the client turnaround time.

    Args:
        response: The server's ``POST /queue`` response (may carry
            ``synthesis_ms`` and ``latency_ms`` when a verbose request).
        script_start: The ``time.monotonic()`` value captured when the CLI
            was launched. Used to compute the turnaround time unless
            ``turnaround_override`` is given.
        turnaround_override: When provided, use this exact turnaround time
            (seconds) instead of computing it from ``script_start``.
            Used by ``speak-region`` to report a turnaround that excludes
            the user's interactive region-selection time.
        breakdown: Optional ordered list of ``(label, seconds)`` pairs to
            print as a per-stage breakdown before the latency summary.

    """
    if breakdown is not None:
        for label, seconds in breakdown:
            typer.echo(f"{label}: {seconds:.3f} s", err=True)
    synthesis_ms = response.get("synthesis_ms")
    latency_ms = response.get("latency_ms")
    synthesis_str = f"{synthesis_ms} ms" if synthesis_ms is not None else "n/a"
    latency_str = f"{latency_ms} ms" if latency_ms is not None else "n/a"
    typer.echo(
        f"Latency: synthesis={synthesis_str}, piper-to-speech={latency_str}",
        err=True,
    )
    if turnaround_override is not None:
        turnaround = turnaround_override
    else:
        turnaround = time.monotonic() - script_start
    typer.echo(f"turnaround-time: {turnaround:.3f} s", err=True)


if __name__ == "__main__":
    app()
