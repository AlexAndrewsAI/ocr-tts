"""Additional coverage tests for the queue client and its helpers."""

import errno
import sys
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from ocr_tts.queue import (
    _find_api_pids,
    _is_connection_refused,
    _launch_server,
    _post_json,
    _signal_pid,
    _version_callback,
    _wait_for_pids_gone,
    _wait_for_server,
    app,
    speak,
)


class TestIsConnectionRefused:
    """Tests for connection-refused classification."""

    def test_direct_refusal(self) -> None:
        """A ConnectionRefusedError reason is classified as refused."""
        exc = urllib.error.URLError(ConnectionRefusedError("refused"))
        assert _is_connection_refused(exc) is True

    def test_oserror_errno_variants(self) -> None:
        """Unreachable-network OSErrors are classified as refused."""
        for code in (errno.ECONNREFUSED, errno.ENETUNREACH, errno.EHOSTUNREACH):
            exc = urllib.error.URLError(OSError(code, "net down"))
            assert _is_connection_refused(exc) is True

    def test_other_oserror_not_refused(self) -> None:
        """An unrelated errno is not treated as refusal."""
        exc = urllib.error.URLError(OSError(errno.EACCES, "denied"))
        assert _is_connection_refused(exc) is False


class TestWaitForServer:
    """Tests for the TCP readiness poller."""

    def test_timeout_returns_false(self) -> None:
        """Persistent connection failures return False after the timeout."""
        with (
            patch(
                "ocr_tts.queue.socket.create_connection",
                side_effect=OSError("refused"),
            ),
            patch("ocr_tts.queue.time.sleep"),
        ):
            assert _wait_for_server("127.0.0.1", 9, timeout=0.05) is False

    def test_success_returns_true(self) -> None:
        """A successful connection returns True immediately."""
        conn = MagicMock()
        with patch("ocr_tts.queue.socket.create_connection", return_value=conn):
            assert _wait_for_server("127.0.0.1", 9) is True
        conn.__exit__.assert_called()


class TestLaunchServer:
    """Tests for the background server launcher."""

    def test_failure_returns_none(self) -> None:
        """Spawn failures are logged and reported as None."""
        with patch(
            "ocr_tts.queue.subprocess.Popen",
            side_effect=OSError("cannot fork"),
        ):
            assert _launch_server("127.0.0.1", 9999) is None

    @patch("ocr_tts.queue.subprocess.Popen")
    def test_spawns_server_with_requested_host_and_port(
        self, mock_popen: MagicMock
    ) -> None:
        """The spawned server argv carries the requested host and port.

        This is the argv the ``python -m ocr_tts.api`` entry point parses,
        so the two must stay in sync (custom ports must not be dropped).
        """
        mock_popen.return_value = MagicMock()
        _launch_server("127.0.0.1", 9000)
        argv = mock_popen.call_args.args[0]
        assert argv == [
            sys.executable,
            "-m",
            "ocr_tts.api",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ]


class TestPostJsonRelaunchFailure:
    """Tests for the relaunch fallback in _post_json."""

    def test_terminates_failed_relaunch_and_exits(self) -> None:
        """When the relaunched server never comes up it is terminated."""
        proc = MagicMock()
        proc.poll.return_value = None
        with (
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError(ConnectionRefusedError("no")),
            ),
            patch("ocr_tts.queue._launch_server", return_value=proc),
            patch("ocr_tts.queue._wait_for_server", return_value=False),
            pytest.raises(typer.Exit),
        ):
            _post_json("http://127.0.0.1:9999/queue", {})
        proc.terminate.assert_called_once()


class TestFindApiPids:
    """Tests for /proc-based server process discovery (M6/M7)."""

    @staticmethod
    def _proc_dir(name: str, cmdline: bytes | Exception) -> MagicMock:
        entry = MagicMock()
        entry.name = name
        if isinstance(cmdline, Exception):
            entry.__truediv__.return_value.read_bytes.side_effect = cmdline
        else:
            entry.__truediv__.return_value.read_bytes.return_value = cmdline
        return entry

    def test_iterdir_oserror_returns_empty(self) -> None:
        """An unreadable /proc yields no pids."""
        path_cls = MagicMock()
        path_cls.return_value.iterdir.side_effect = OSError("no /proc")
        with patch("ocr_tts.queue.Path", path_cls):
            assert _find_api_pids("127.0.0.1", 8000) == []

    def test_unreadable_cmdline_is_skipped(self) -> None:
        """Entries whose cmdline cannot be read are skipped."""
        good = self._proc_dir(
            "123", b"python\0-m\0ocr_tts.api\x00--host\x00127.0.0.1\0--port\08000\0"
        )
        bad = self._proc_dir("456", OSError("vanished"))
        non_numeric = MagicMock()
        non_numeric.name = "self"
        path_cls = MagicMock()
        path_cls.return_value.iterdir.return_value = [good, bad, non_numeric]
        with patch("ocr_tts.queue.Path", path_cls):
            assert _find_api_pids("127.0.0.1", 8000) == [123]

    def test_no_match_yields_empty(self) -> None:
        """Processes without matching cmdlines are ignored."""
        other = self._proc_dir("123", b"python\0-m\0othermod\0")
        path_cls = MagicMock()
        path_cls.return_value.iterdir.return_value = [other]
        with patch("ocr_tts.queue.Path", path_cls):
            assert _find_api_pids("127.0.0.1", 8000) == []

    def test_port_substring_is_not_matched(self) -> None:
        """Port 18000 must not match a search for port 8000 (M6)."""
        wrong_port = self._proc_dir(
            "123", b"python\0-m\0ocr_tts.api\x00--host\x00127.0.0.1\0--port\018000\0"
        )
        path_cls = MagicMock()
        path_cls.return_value.iterdir.return_value = [wrong_port]
        with patch("ocr_tts.queue.Path", path_cls):
            assert _find_api_pids("127.0.0.1", 8000) == []

    def test_host_mismatch_is_not_matched(self) -> None:
        """A different --host is not matched (M7)."""
        other_host = self._proc_dir(
            "123", b"python\0-m\0ocr_tts.api\x00--host\x000.0.0.0\0--port\08000\0"
        )
        path_cls = MagicMock()
        path_cls.return_value.iterdir.return_value = [other_host]
        with patch("ocr_tts.queue.Path", path_cls):
            assert _find_api_pids("127.0.0.1", 8000) == []

    def test_multiple_sorted_pids(self) -> None:
        """All matching servers are returned in sorted order."""
        ent = []
        for name in ("30", "10", "20"):
            ent.append(
                self._proc_dir(
                    name,
                    b"python\0-m\0ocr_tts.api\x00--host\x00127.0.0.1\0--port\08000\0",
                )
            )
        path_cls = MagicMock()
        path_cls.return_value.iterdir.return_value = ent
        with patch("ocr_tts.queue.Path", path_cls):
            assert _find_api_pids("127.0.0.1", 8000) == [10, 20, 30]


class TestSignalPid:
    """Tests for pid signalling robustness."""

    def test_process_lookup_error_is_swallowed(self) -> None:
        """Vanishing processes during group signalling are tolerated."""
        with (
            patch("os.getpgid", return_value=777),
            patch("os.killpg", side_effect=ProcessLookupError("gone")),
        ):
            _signal_pid(777, 15)

    def test_permission_error_is_swallowed(self) -> None:
        """Unkillable processes are tolerated."""
        with (
            patch("os.getpgid", return_value=1),
            patch("os.kill", side_effect=PermissionError("nope")),
        ):
            _signal_pid(4242, 15)

    def test_killpg_used_for_group_leaders(self) -> None:
        """Group leaders are killed via their process group."""
        with (
            patch("os.getpgid", return_value=777),
            patch("os.killpg") as killpg,
        ):
            _signal_pid(777, 15)
        killpg.assert_called_once_with(777, 15)


class TestWaitForPidsGone:
    """Tests for the reap-wait helper."""

    def test_sleeps_while_pids_alive(self) -> None:
        """The wait loop polls until pids disappear or deadline passes."""
        with (
            patch("ocr_tts.queue._pid_exists", return_value=True),
            patch("ocr_tts.queue.time.sleep") as sleep,
        ):
            _wait_for_pids_gone([1], timeout=0.06)
        assert sleep.called


class TestVersionCallback:
    """Tests for the speak CLI version plumbing."""

    def test_version_callback_prints_and_exits(self) -> None:
        """The eager version callback prints and raises Exit."""
        with pytest.raises(typer.Exit):
            _version_callback(True)

    def test_version_callback_falsy_is_noop(self) -> None:
        """A falsy value leaves the callback inert."""
        _version_callback(False)

    def test_cli_version_flag(self, runner: CliRunner) -> None:
        """--version prints the speak version and exits cleanly."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "speak" in result.output


class TestSpeakVersionShortCircuit:
    """Ensure a set --version option short-circuits speak()."""

    def test_speak_with_version_flag_returns_early(self) -> None:
        """speak() called with version=True returns before validation."""
        speak(text="hi", version=True)


class TestVersionResolution:
    """Tests for guarded import-time version resolution (M9)."""

    def test_falls_back_when_distribution_missing(self) -> None:
        """A missing distribution yields the default version, not a crash."""
        import importlib

        import ocr_tts._version as version_mod

        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("no dist"),
        ):
            reloaded = importlib.reload(version_mod)
        assert reloaded.__version__ == "0.1.0"

    def test_uses_distribution_version_when_available(self) -> None:
        """An installed distribution supplies the real metadata version."""
        import importlib

        import ocr_tts._version as version_mod

        with patch("importlib.metadata.version", return_value="9.9.9"):
            reloaded = importlib.reload(version_mod)
        assert reloaded.__version__ == "9.9.9"


class TestSendClearRequest:
    """Tests for the queue-clear client helper (M5)."""

    def test_posts_to_clear_endpoint(self) -> None:
        """The helper posts an empty payload to /queue/clear."""
        from ocr_tts.queue import send_clear_request

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "cleared", "queue_size": 0}'
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = mock_response
            result = send_clear_request(host="h", port=1)

        assert result == {"status": "cleared", "queue_size": 0}
        request = urlopen.call_args.args[0]
        assert request.full_url.endswith("/queue/clear")

    def test_connection_refused_returns_none_without_launching(self) -> None:
        """A refused connection returns None and never auto-launches."""
        from ocr_tts.queue import send_clear_request

        with (
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError(ConnectionRefusedError("no")),
            ),
            patch("ocr_tts.queue._launch_server") as launch,
        ):
            assert send_clear_request(host="h", port=1) is None
        launch.assert_not_called()

    def test_non_connection_urlerror_exits(self) -> None:
        """A non-refused URL error escalates to a non-zero exit."""
        from ocr_tts.queue import send_clear_request

        with (
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.URLError("dns failure"),
            ),
            pytest.raises(typer.Exit),
        ):
            send_clear_request(host="h", port=1)
