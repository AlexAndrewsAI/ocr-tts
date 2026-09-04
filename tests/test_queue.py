"""Tests for the speak CLI client that controls the TTS server queue."""

import io
import json
import logging
import signal
import subprocess
import sys
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from ocr_tts import queue as queue_module
from ocr_tts.cli import app as cli_app
from ocr_tts.queue import app


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Typer CLI test runner."""
    return CliRunner()


class TestSpeakCLI:
    """Tests for the speak command."""

    def test_app_is_typer(self) -> None:
        """The queue app is a Typer instance."""
        assert isinstance(app, typer.Typer)

    @patch("ocr_tts.queue.send_speak_request")
    def test_speak_enqueues_text(self, mock_send: MagicMock, runner: CliRunner) -> None:
        """Speak POSTs the text to the server queue with defaults."""
        mock_send.return_value = {"status": "queued", "queue_size": 2}
        result = runner.invoke(app, ["Hello world"])
        assert result.exit_code == 0
        mock_send.assert_called_once_with(
            "Hello world",
            host="127.0.0.1",
            port=8000,
            voice="en_US-hfc_male-medium",
            speed=1.0,
            verbose=False,
        )
        assert "2" in result.output

    @patch("ocr_tts.queue.send_speak_request")
    def test_speak_custom_voice_speed_host(
        self, mock_send: MagicMock, runner: CliRunner
    ) -> None:
        """Speak forwards explicit voice/speed/host/port options."""
        mock_send.return_value = {"status": "queued", "queue_size": 1}
        result = runner.invoke(
            app,
            [
                "Bonjour",
                "-v",
                "fr_FR-siwis-medium",
                "-s",
                "1.2",
                "--host",
                "localhost",
                "--port",
                "9000",
            ],
        )
        assert result.exit_code == 0
        mock_send.assert_called_once_with(
            "Bonjour",
            host="localhost",
            port=9000,
            voice="fr_FR-siwis-medium",
            speed=1.2,
            verbose=False,
        )

    @patch("ocr_tts.queue.send_clear_request")
    def test_speak_no_longer_accepts_clear(
        self, mock_clear: MagicMock, runner: CliRunner
    ) -> None:
        """--clear is no longer an option on speak; use `api clear` instead."""
        result = runner.invoke(app, ["hello", "--clear"])
        assert result.exit_code == 2
        mock_clear.assert_not_called()
        assert "No such option" in result.output
        assert "--clear" in result.output

    @patch("ocr_tts.queue.send_speak_request")
    def test_speak_missing_text_errors(
        self, mock_send: MagicMock, runner: CliRunner
    ) -> None:
        """Speak without text exits with an error."""
        result = runner.invoke(app, [])
        assert result.exit_code == 1
        assert "TEXT is required" in result.output
        mock_send.assert_not_called()

    @patch("ocr_tts.queue.send_speak_request")
    def test_speak_blank_text_errors(
        self, mock_send: MagicMock, runner: CliRunner
    ) -> None:
        """Speak with only whitespace text exits with an error."""
        result = runner.invoke(app, ["   "])
        assert result.exit_code == 1
        mock_send.assert_not_called()

    @patch("ocr_tts.queue.send_speak_request")
    def test_speak_verbose_reports_latency_and_turnaround(
        self, mock_send: MagicMock, runner: CliRunner
    ) -> None:
        """--verbose forwards wait=True and prints latency + turnaround-time."""
        mock_send.return_value = {
            "status": "queued",
            "queue_size": 1,
            "synthesis_ms": 12.5,
            "latency_ms": 34.2,
        }
        result = runner.invoke(app, ["hello", "--verbose"])
        assert result.exit_code == 0
        mock_send.assert_called_once_with(
            "hello",
            host="127.0.0.1",
            port=8000,
            voice="en_US-hfc_male-medium",
            speed=1.0,
            verbose=True,
        )
        assert "Latency: synthesis=12.5 ms" in result.output
        assert "piper-to-speech=34.2 ms" in result.output
        assert "turnaround-time:" in result.output


class TestHTTPErrors:
    """Tests for server connection/error handling."""

    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_server_error_exits(
        self, mock_urlopen: MagicMock, runner: CliRunner
    ) -> None:
        """An HTTP error response produces a friendly exit message."""
        exc = urllib.error.HTTPError(
            "url", 500, "Server Error", Message(), io.BytesIO(b"boom")
        )
        mock_urlopen.side_effect = exc
        result = runner.invoke(app, ["hello"])
        assert result.exit_code == 1
        assert "Server error (500)" in result.output
        assert "boom" in result.output

    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_connection_error_exits(
        self, mock_urlopen: MagicMock, runner: CliRunner
    ) -> None:
        """An unreachable server produces a friendly exit message."""
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        result = runner.invoke(app, ["hello"])
        assert result.exit_code == 1
        assert "Cannot reach TTS server" in result.output

    @patch("ocr_tts.queue.subprocess.Popen")
    @patch("ocr_tts.queue._wait_for_server", return_value=False)
    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_connection_refused_server_fails_to_start(
        self,
        mock_urlopen: MagicMock,
        _mock_wait: MagicMock,
        mock_popen: MagicMock,
        runner: CliRunner,
    ) -> None:
        """If the server cannot be launched, exit with a clear message."""
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        mock_popen.return_value = MagicMock()
        result = runner.invoke(app, ["hello"])
        assert result.exit_code == 1
        assert "Cannot reach TTS server" in result.output
        assert "Failed to launch TTS API server" in result.output
        mock_popen.assert_called_once()
        mock_urlopen.assert_called_once()

    @patch("ocr_tts.queue.subprocess.Popen")
    @patch("ocr_tts.queue._wait_for_server", return_value=True)
    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_connection_refused_launches_server_and_retries(
        self,
        mock_urlopen: MagicMock,
        _mock_wait: MagicMock,
        mock_popen: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Connection refused launches the server and retries successfully."""
        success_response = MagicMock()
        success_response.read.return_value = b'{"status": "queued", "queue_size": 1}'
        success_cm = MagicMock()
        success_cm.__enter__ = MagicMock(return_value=success_response)
        success_cm.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.URLError("connection refused"),
            success_cm,
        ]
        mock_popen.return_value = MagicMock()

        result = runner.invoke(app, ["hello"])

        assert result.exit_code == 0
        mock_popen.assert_called_once()
        assert mock_urlopen.call_count == 2
        assert "launch" in result.output.lower()

    @patch("ocr_tts.queue.subprocess.Popen")
    @patch("ocr_tts.queue._wait_for_server", return_value=True)
    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_connection_refused_second_time_is_hard_error(
        self,
        mock_urlopen: MagicMock,
        _mock_wait: MagicMock,
        mock_popen: MagicMock,
        runner: CliRunner,
    ) -> None:
        """A second connection refused is a hard error after one retry."""
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        mock_popen.return_value = MagicMock()

        result = runner.invoke(app, ["hello"])

        assert result.exit_code == 1
        assert "Cannot reach TTS server" in result.output
        mock_popen.assert_called_once()
        assert mock_urlopen.call_count == 2

    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_non_connection_refused_not_retried(
        self, mock_urlopen: MagicMock, runner: CliRunner
    ) -> None:
        """URLErrors that are not connection-refused are not retried."""
        mock_urlopen.side_effect = urllib.error.URLError("some other network error")
        result = runner.invoke(app, ["hello"])
        assert result.exit_code == 1
        assert "Cannot reach TTS server" in result.output
        mock_urlopen.assert_called_once()

    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_post_json_sends_payload(self, mock_urlopen: MagicMock) -> None:
        """send_speak_request POSTs a JSON body to the queue endpoint."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "queued", "queue_size": 3}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = queue_module.send_speak_request(
            "hi", host="h", port=1234, voice="v", speed=2.0
        )

        assert result == {"status": "queued", "queue_size": 3}
        # A successful request must be sent exactly once (regression: the
        # retry loop previously fell through and re-sent the POST twice).
        assert mock_urlopen.call_count == 1
        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "http://h:1234/queue"
        assert request.method == "POST"
        assert json.loads(request.data) == {
            "text": "hi",
            "voice": "v",
            "speed": 2.0,
        }
        assert request.get_header("Content-type") == "application/json"

    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_post_json_verbose_adds_wait(self, mock_urlopen: MagicMock) -> None:
        """Verbose requests include ``wait: true`` in the queue payload."""
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'{"status": "queued", "queue_size": 0, "synthesis_ms": 1.0}'
        )
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = queue_module.send_speak_request(
            "hi", host="h", port=1234, voice="v", speed=1.0, verbose=True
        )

        assert result["synthesis_ms"] == 1.0
        request = mock_urlopen.call_args[0][0]
        assert json.loads(request.data) == {
            "text": "hi",
            "voice": "v",
            "speed": 1.0,
            "wait": True,
        }

        assert request.get_header("Content-type") == "application/json"

    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_post_json_success_sent_once(self, mock_urlopen: MagicMock) -> None:
        """A successful POST must not be re-sent by the retry loop.

        Regression test for the double-speech bug: the retry loop had no
        break after a successful request, so it sent every message twice.
        """
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "queued", "queue_size": 1}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = queue_module._post_json("http://h:1/queue", {"text": "hi"})

        assert result == {"status": "queued", "queue_size": 1}
        mock_urlopen.assert_called_once()

    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_post_json_connection_refused_retries_once(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Connection refusal triggers a single retry, not a duplicate send."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "queued", "queue_size": 1}'
        success_cm = MagicMock()
        success_cm.__enter__ = MagicMock(return_value=mock_response)
        success_cm.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            urllib.error.URLError("connection refused"),
            success_cm,
        ]

        with (
            patch("ocr_tts.queue._launch_server", return_value=MagicMock()),
            patch("ocr_tts.queue._wait_for_server", return_value=True),
        ):
            result = queue_module._post_json("http://h:1/queue", {"text": "hi"})

        assert result == {"status": "queued", "queue_size": 1}
        assert mock_urlopen.call_count == 2


class TestCloseProcessHelpers:
    """Tests for the process discovery / termination helpers."""

    def test_find_api_pids_matches_module_and_port(self, tmp_path: Path) -> None:
        """Only processes running ocr_tts.api with the host/port pair match."""
        (tmp_path / "1234").mkdir()
        (tmp_path / "1234" / "cmdline").write_bytes(
            b"python\x00-m\x00ocr_tts.api\x00--host\x00127.0.0.1\x00--port\x008000\x00"
        )
        # Same module, different port -> excluded.
        (tmp_path / "5678").mkdir()
        (tmp_path / "5678" / "cmdline").write_bytes(
            b"python\x00-m\x00ocr_tts.api\x00--host\x00127.0.0.1\x00--port\x009000\x00"
        )
        # Same port, different module -> excluded.
        (tmp_path / "9999").mkdir()
        (tmp_path / "9999" / "cmdline").write_bytes(
            b"python\x00-m\x00something_else\x00--port\x008000\x00"
        )
        # Non-numeric directory -> skipped.
        (tmp_path / "notnum").mkdir()

        with patch("ocr_tts.queue.Path", return_value=tmp_path):
            assert queue_module._find_api_pids("127.0.0.1", 8000) == [1234]

    @patch("ocr_tts.queue.os.kill", side_effect=ProcessLookupError)
    def test_pid_exists_missing(self, mock_kill: MagicMock) -> None:
        """A non-existent process is reported as not running."""
        assert queue_module._pid_exists(1234) is False
        mock_kill.assert_called_once_with(1234, 0)

    @patch("ocr_tts.queue.os.kill", side_effect=PermissionError)
    def test_pid_exists_permission_denied_treated_as_alive(
        self, mock_kill: MagicMock
    ) -> None:
        """PermissionError means the process exists but we cannot signal it."""
        assert queue_module._pid_exists(1) is True
        mock_kill.assert_called_once_with(1, 0)

    @patch("ocr_tts.queue.os.kill")
    @patch("ocr_tts.queue.os.killpg")
    @patch("ocr_tts.queue.os.getpgid", return_value=2222)
    def test_signal_pid_uses_group_when_leader(
        self, mock_getpgid: MagicMock, mock_killpg: MagicMock, mock_kill: MagicMock
    ) -> None:
        """A process-group leader is signalled via the whole group."""
        queue_module._signal_pid(2222, signal.SIGTERM)
        mock_getpgid.assert_called_once_with(2222)
        mock_killpg.assert_called_once_with(2222, signal.SIGTERM)
        mock_kill.assert_not_called()

    @patch("ocr_tts.queue.os.kill")
    @patch("ocr_tts.queue.os.killpg")
    @patch("ocr_tts.queue.os.getpgid", return_value=1111)
    def test_signal_pid_falls_back_to_pid_when_not_leader(
        self, mock_getpgid: MagicMock, mock_killpg: MagicMock, mock_kill: MagicMock
    ) -> None:
        """A non-leader process is signalled individually (never a group)."""
        queue_module._signal_pid(2222, signal.SIGTERM)
        mock_getpgid.assert_called_once_with(2222)
        mock_kill.assert_called_once_with(2222, signal.SIGTERM)
        mock_killpg.assert_not_called()

    def test_terminate_pid_kills_real_process(self) -> None:
        """_terminate_pid shuts down a live process and reaps it."""
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert child.poll() is None
            queue_module._terminate_pid(child.pid)
            child.wait(timeout=10)
            assert child.returncode is not None
        finally:
            if child.poll() is None:
                child.kill()

    def test_wait_for_pids_gone_returns_when_gone(self) -> None:
        """_wait_for_pids_gone returns immediately when no process remains."""
        with patch("ocr_tts.queue._pid_exists", return_value=False):
            queue_module._wait_for_pids_gone([1, 2])


class TestCloseCLI:
    """Tests for `ocr-tts api close`, which tears down the TTS server."""

    @patch(
        "ocr_tts.queue.send_shutdown_request",
        return_value={"status": "shutting_down"},
    )
    @patch("ocr_tts.queue._pid_exists", return_value=False)
    @patch("ocr_tts.queue._wait_for_pids_gone")
    @patch("ocr_tts.queue._find_api_pids", return_value=[1234, 5678])
    @patch("ocr_tts.queue._terminate_pid")
    def test_close_sends_shutdown_command(
        self,
        mock_terminate: MagicMock,
        mock_find: MagicMock,
        mock_wait: MagicMock,
        mock_pid_exists: MagicMock,
        mock_send: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Close sends the shutdown command, then waits for server exit."""
        result = runner.invoke(cli_app, ["api", "close"])
        assert result.exit_code == 0
        mock_send.assert_called_once_with(host="127.0.0.1", port=8000)
        mock_find.assert_called_once_with("127.0.0.1", 8000)
        mock_wait.assert_called_once()
        # No stragglers remain -> nothing is force-killed.
        assert mock_pid_exists.call_count == 2
        mock_terminate.assert_not_called()
        assert "Sent shutdown command" in result.output
        assert "TTS API server closed." in result.output

    def test_close_logs_matched_pids(
        self,
        caplog: pytest.LogCaptureFixture,
        runner: CliRunner,
    ) -> None:
        """Close logs the matched server PIDs before signalling (M6)."""
        with (
            patch(
                "ocr_tts.queue.send_shutdown_request",
                return_value={"status": "shutting_down"},
            ),
            patch(
                "ocr_tts.queue._find_api_pids",
                return_value=[1234, 5678],
            ) as mock_find,
            caplog.at_level(logging.INFO, logger="ocr_tts.queue"),
        ):
            result = runner.invoke(cli_app, ["api", "close"])
        assert result.exit_code == 0
        mock_find.assert_called_once_with("127.0.0.1", 8000)
        assert "1234" in caplog.text
        assert "5678" in caplog.text

    @patch(
        "ocr_tts.queue.send_shutdown_request",
        return_value={"status": "shutting_down"},
    )
    @patch("ocr_tts.queue._pid_exists", return_value=False)
    @patch("ocr_tts.queue._wait_for_pids_gone")
    @patch("ocr_tts.queue._find_api_pids", return_value=[1234])
    @patch("ocr_tts.queue._terminate_pid")
    def test_close_custom_port(
        self,
        mock_terminate: MagicMock,
        mock_find: MagicMock,
        mock_wait: MagicMock,
        mock_pid_exists: MagicMock,
        mock_send: MagicMock,
        runner: CliRunner,
    ) -> None:
        """A custom --port is forwarded to the shutdown request."""
        result = runner.invoke(cli_app, ["api", "close", "--port", "9000"])
        assert result.exit_code == 0
        mock_send.assert_called_once_with(host="127.0.0.1", port=9000)
        mock_find.assert_called_once_with("127.0.0.1", 9000)
        mock_wait.assert_called_once()
        assert mock_pid_exists.call_count == 1
        mock_terminate.assert_not_called()

    @patch("ocr_tts.queue.send_shutdown_request", return_value=None)
    @patch("ocr_tts.queue._find_api_pids")
    def test_close_no_server_is_noop(
        self, mock_find: MagicMock, mock_send: MagicMock, runner: CliRunner
    ) -> None:
        """Close is a no-op when no server is running."""
        result = runner.invoke(cli_app, ["api", "close"])
        assert result.exit_code == 0
        mock_send.assert_called_once_with(host="127.0.0.1", port=8000)
        mock_find.assert_not_called()
        assert "No running TTS API server" in result.output
        assert "Nothing to close." in result.output

    @patch(
        "ocr_tts.queue.send_shutdown_request",
        return_value={"status": "shutting_down"},
    )
    @patch("ocr_tts.queue._pid_exists", return_value=True)
    @patch("ocr_tts.queue._wait_for_pids_gone")
    @patch("ocr_tts.queue._find_api_pids", return_value=[1234])
    @patch("ocr_tts.queue._terminate_pid")
    def test_close_force_kills_stragglers(
        self,
        mock_terminate: MagicMock,
        mock_find: MagicMock,
        mock_wait: MagicMock,
        mock_pid_exists: MagicMock,
        mock_send: MagicMock,
        runner: CliRunner,
    ) -> None:
        """A server still running after the command is force-terminated."""
        result = runner.invoke(cli_app, ["api", "close"])
        assert result.exit_code == 0
        mock_send.assert_called_once_with(host="127.0.0.1", port=8000)
        mock_find.assert_called_once_with("127.0.0.1", 8000)
        mock_wait.assert_called_once()
        assert mock_pid_exists.call_count == 1
        mock_terminate.assert_called_once_with(1234)
        assert "TTS API server closed." in result.output

    @patch(
        "ocr_tts.queue.send_shutdown_request",
        return_value={"status": "shutting_down"},
    )
    @patch("ocr_tts.queue._pid_exists", return_value=True)
    @patch("ocr_tts.queue._wait_for_pids_gone")
    @patch("ocr_tts.queue._find_api_pids", return_value=[1234])
    @patch("ocr_tts.queue._terminate_pid", side_effect=PermissionError("denied"))
    def test_close_permission_error_exits(
        self,
        mock_terminate: MagicMock,
        mock_find: MagicMock,
        mock_wait: MagicMock,
        mock_pid_exists: MagicMock,
        mock_send: MagicMock,
        runner: CliRunner,
    ) -> None:
        """A PermissionError while force-killing escalates to a non-zero exit."""
        result = runner.invoke(cli_app, ["api", "close"])
        assert result.exit_code == 1
        mock_send.assert_called_once_with(host="127.0.0.1", port=8000)
        mock_find.assert_called_once_with("127.0.0.1", 8000)
        mock_wait.assert_called_once()
        assert mock_pid_exists.call_count == 1
        mock_terminate.assert_called_once_with(1234)
        assert "Permission denied closing process" in result.output


class TestSendShutdownRequest:
    """Tests for the client-side shutdown command (api close)."""

    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_success_returns_response(self, mock_urlopen: MagicMock) -> None:
        """A running server returns its shutdown confirmation."""
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "shutting_down"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = queue_module.send_shutdown_request(host="h", port=1234)

        assert result == {"status": "shutting_down"}
        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "http://h:1234/shutdown"
        assert request.method == "POST"
        assert request.get_header("Content-type") == "application/json"

    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_connection_refused_returns_none(self, mock_urlopen: MagicMock) -> None:
        """A refused connection means no server is running -> None."""
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        assert queue_module.send_shutdown_request() is None

    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_other_urlerror_exits(self, mock_urlopen: MagicMock) -> None:
        """A non-refused network error is a hard failure."""
        mock_urlopen.side_effect = urllib.error.URLError("some other network error")
        with pytest.raises(typer.Exit) as exc_info:
            queue_module.send_shutdown_request(host="h", port=1)
        assert exc_info.value.exit_code == 1

    @patch("ocr_tts.queue.urllib.request.urlopen")
    def test_http_error_exits(self, mock_urlopen: MagicMock) -> None:
        """A server HTTP error is surfaced via typer.Exit."""
        http_error = urllib.error.HTTPError(
            "http://h:1/shutdown", 500, "err", Message(), io.BytesIO(b"boom")
        )
        mock_urlopen.side_effect = http_error
        with pytest.raises(typer.Exit) as exc_info:
            queue_module.send_shutdown_request(host="h", port=1)
        assert exc_info.value.exit_code == 1


class TestClearCLI:
    """Tests for `ocr-tts api clear`, which wipes the running server queue."""

    @patch("ocr_tts.queue.send_clear_request")
    def test_clear_wipes_queue(self, mock_clear: MagicMock, runner: CliRunner) -> None:
        """`api clear` POSTs to /queue/clear and reports the queue size."""
        mock_clear.return_value = {"status": "cleared", "queue_size": 0}
        result = runner.invoke(cli_app, ["api", "clear"])
        assert result.exit_code == 0
        mock_clear.assert_called_once_with(host="127.0.0.1", port=8000)
        assert "Queue cleared; 0 item(s) pending" in result.output

    @patch("ocr_tts.queue.send_clear_request")
    def test_clear_forwards_host_port(
        self, mock_clear: MagicMock, runner: CliRunner
    ) -> None:
        """`api clear` forwards --host/--port to the clear request."""
        mock_clear.return_value = {"status": "cleared", "queue_size": 3}
        result = runner.invoke(
            cli_app, ["api", "clear", "--host", "h", "--port", "9000"]
        )
        assert result.exit_code == 0
        mock_clear.assert_called_once_with(host="h", port=9000)
        assert "Queue cleared; 3 item(s) pending" in result.output

    @patch("ocr_tts.queue.send_clear_request", return_value=None)
    def test_clear_with_no_server_reports_already_cleared(
        self, mock_clear: MagicMock, runner: CliRunner
    ) -> None:
        """`api clear` with no server running does not launch one (M5)."""
        result = runner.invoke(cli_app, ["api", "clear"])
        assert result.exit_code == 0
        mock_clear.assert_called_once_with(host="127.0.0.1", port=8000)
        assert "Queue is already cleared." in result.output
