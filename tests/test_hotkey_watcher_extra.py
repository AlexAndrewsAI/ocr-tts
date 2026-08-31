"""Additional coverage tests for the hotkey watcher."""

import logging
import sys
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from ocr_tts.hotkey_watcher import (
    HotkeyAction,
    HotkeyConfig,
    HotkeyConfigItem,
    HotkeyWatcher,
    app,
    create_default_config,
    execute_action,
    get_default_config_path,
    load_config,
    run_ocr_region,
)


@pytest.fixture
def speak_item() -> HotkeyConfigItem:
    """Provide a basic speak-text binding."""
    return HotkeyConfigItem(hotkey="<ctrl>+<shift>+s", action=HotkeyAction.SPEAK_TEXT)


class TestRunOcrRegionBlankWarning:
    """Tests for the blank-capture warning in run_ocr_region."""

    def test_blank_capture_warns_but_queues(
        self, speak_item: HotkeyConfigItem, caplog: Any
    ) -> None:
        """A blank capture logs a warning and still queues detected text."""
        with (
            patch(
                "ocr_tts.ocr_region.select_region",
                return_value=MagicMock(width=10, height=10),
            ),
            patch("ocr_tts.ocr_region.capture_selected_region", return_value="img"),
            patch("ocr_tts.ocr_region.image_is_blank", return_value=True),
            patch("ocr_tts.ocr_region.extract_text", return_value="words"),
            patch(
                "ocr_tts.hotkey_watcher.send_speak_request",
                return_value={"queue_size": 1},
            ) as send,
            caplog.at_level(logging.WARNING, logger="ocr_tts.hotkey_watcher"),
        ):
            result = run_ocr_region(speak_item)
        assert "blank" in caplog.text
        assert result == {"status": "ok", "queue_size": 1}
        send.assert_called_once()


class TestExecuteActionEdgeCases:
    """Tests for execute_action's defensive branches."""

    def test_unknown_action_reports_error(self) -> None:
        """An unrecognized action yields an error result."""
        item = HotkeyConfigItem(
            hotkey="<ctrl>+<shift>+z", action=HotkeyAction.SPEAK_TEXT
        )
        item.action = "not-a-real-action"  # type: ignore[assignment]
        result = execute_action(item)
        assert result["status"] == "error"
        assert "unknown action" in str(result["reason"])

    def test_system_exit_propagates(self, speak_item: HotkeyConfigItem) -> None:
        """SystemExit from the queue client is re-raised."""
        with (
            patch(
                "ocr_tts.hotkey_watcher.send_speak_request", side_effect=SystemExit(1)
            ),
            pytest.raises(SystemExit),
        ):
            execute_action(speak_item)


class _FakeListener:
    """Stand-in for pynput's GlobalHotKeys listener."""

    def __init__(self, callbacks: dict[str, Any]) -> None:
        self.callbacks = callbacks
        self.started = False
        self.stopped = False
        self.join_side_effect: BaseException | None = None

    def start(self) -> None:
        self.started = True

    def join(self) -> None:
        if self.join_side_effect is not None:
            raise self.join_side_effect

    def is_alive(self) -> bool:
        return self.started and not self.stopped

    def stop(self) -> None:
        self.stopped = True


def _install_pynput_mock(
    join_raises: BaseException | None = None,
) -> list[_FakeListener]:
    """Install a mocked pynput module and return created listeners."""
    listeners: list[_FakeListener] = []

    def factory(callbacks: dict[str, Any]) -> _FakeListener:
        listener = _FakeListener(callbacks)
        listener.join_side_effect = join_raises
        listeners.append(listener)
        return listener

    keyboard_mod = SimpleNamespace(GlobalHotKeys=factory)
    pynput_mod = SimpleNamespace(keyboard=keyboard_mod)
    sys.modules["pynput"] = pynput_mod  # type: ignore[assignment]
    sys.modules["pynput.keyboard"] = keyboard_mod  # type: ignore[assignment]
    return listeners


def _uninstall_pynput_mock() -> None:
    """Remove the mocked pynput modules."""
    sys.modules.pop("pynput", None)
    sys.modules.pop("pynput.keyboard", None)


class TestHotkeyWatcherLifecycle:
    """Tests for HotkeyWatcher start/stop behaviour."""

    def test_start_runs_and_stops_listener(self, caplog: Any) -> None:
        """start() blocks on the listener and stops it in finally."""
        listeners = _install_pynput_mock()
        try:
            watcher = HotkeyWatcher(create_default_config())
            with caplog.at_level(logging.INFO, logger="ocr_tts.hotkey_watcher"):
                watcher.start()
            assert len(listeners) == 1
            assert listeners[0].started
            assert listeners[0].stopped
            assert watcher.running is False
            assert "Starting hotkey watcher" in caplog.text
        finally:
            _uninstall_pynput_mock()

    def test_start_handles_keyboard_interrupt(self, caplog: Any) -> None:
        """Ctrl+C during listening stops the watcher gracefully."""
        listeners = _install_pynput_mock(join_raises=KeyboardInterrupt())
        try:
            watcher = HotkeyWatcher(create_default_config())
            with caplog.at_level(logging.INFO, logger="ocr_tts.hotkey_watcher"):
                watcher.start()
            assert "interrupted by user" in caplog.text
            assert watchers_clean(watcher)
        finally:
            _uninstall_pynput_mock()
            del listeners

    def test_stop_twice_is_safe(self) -> None:
        """stop() without a listener is a no-op."""
        watcher = HotkeyWatcher(HotkeyConfig())
        watcher.stop()
        assert watcher.running is False


def watchers_clean(watcher: HotkeyWatcher) -> bool:
    """Return True when the watcher no longer holds a listener."""
    return watcher._listener is None


class TestStartCommand:
    """Tests for the hotkey-watcher start command paths."""

    def test_start_success_echoes_running(
        self, runner: CliRunner, tmp_path: Any
    ) -> None:
        """A valid config starts the watcher and reports it running."""
        config_file = tmp_path / "hotkeys.yaml"
        config_file.write_text(
            "hotkeys:\n  - hotkey: '<ctrl>+<shift>+s'\n    action: speak-text\n"
        )
        with patch.object(HotkeyWatcher, "start", return_value=None):
            result = runner.invoke(app, ["start", "--config", str(config_file)])
        assert result.exit_code == 0
        assert "running" in result.output

    def test_start_invalid_config_fails(self, runner: CliRunner, tmp_path: Any) -> None:
        """A schema-violating config exits with an error message."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(
            "hotkeys:\n  - hotkey: '<ctrl>+<shift>+s'\n    action: bogus-action\n"
        )
        result = runner.invoke(app, ["start", "--config", str(config_file)])
        assert result.exit_code == 1
        assert "Configuration error" in result.output

    def test_start_runtime_failure_fails(
        self, runner: CliRunner, tmp_path: Any
    ) -> None:
        """A failing listener startup exits with code 1."""
        config_file = tmp_path / "hotkeys.yaml"
        config_file.write_text(
            "hotkeys:\n  - hotkey: '<ctrl>+<shift>+s'\n    action: speak-text\n"
        )
        with patch.object(
            HotkeyWatcher, "start", side_effect=RuntimeError("no display")
        ):
            result = runner.invoke(app, ["start", "--config", str(config_file)])
        assert result.exit_code == 1
        assert "Failed to start hotkey watcher" in result.output


class TestDispatchUiExecutor:
    """Ensure UI actions share the single-worker executor."""

    def test_ui_action_submitted_to_executor(self) -> None:
        """UI actions are routed through the shared executor."""
        import ocr_tts.hotkey_watcher as hw

        item = HotkeyConfigItem(
            hotkey="<ctrl>+<shift>+o", action=HotkeyAction.SEND_REGION
        )
        done = threading.Event()

        def fake_execute(_item: HotkeyConfigItem) -> dict[str, Any]:
            done.set()
            return {"status": "ok"}

        with patch.object(hw, "execute_action", side_effect=fake_execute):
            hw.dispatch_action(item)
            assert done.wait(5.0)


class TestStartCommandDefaultConfig:
    """Tests for the default-config path of the start command."""

    def test_start_without_config_uses_example_yaml(
        self, runner: CliRunner, tmp_path: Any
    ) -> None:
        """With no --config, the bundled example config is loaded."""
        example = tmp_path / "example.yaml"
        example.write_text(
            "hotkeys:\n  - hotkey: '<ctrl>+<shift>+s'\n    action: speak-text\n"
        )
        with (
            patch(
                "ocr_tts.hotkey_watcher.get_default_config_path",
                return_value=example,
            ),
            patch.object(HotkeyWatcher, "start", return_value=None),
        ):
            result = runner.invoke(app, ["start"])
        assert result.exit_code == 0


class TestLaunchAction:
    """Tests for the launch (api launch) hotkey action."""

    def _item(self) -> HotkeyConfigItem:
        """Provide a launch binding."""
        return HotkeyConfigItem(hotkey="<ctrl>+<shift>+l", action=HotkeyAction.LAUNCH)

    def test_launch_success_when_server_ready(self) -> None:
        """A spawned server that becomes ready reports success."""
        with (
            patch("ocr_tts.queue._launch_server", return_value=MagicMock()) as launch,
            patch("ocr_tts.queue._wait_for_server", return_value=True),
        ):
            result = execute_action(self._item())
        assert result == {"status": "ok"}
        launch.assert_called_once_with("127.0.0.1", 8000)

    def test_launch_failure_when_spawn_fails(self) -> None:
        """A failed spawn reports an error."""
        with (
            patch("ocr_tts.queue._launch_server", return_value=None),
            patch("ocr_tts.queue._wait_for_server") as wait,
        ):
            result = execute_action(self._item())
        assert result["status"] == "error"
        assert "failed to start" in str(result["reason"])
        wait.assert_not_called()

    def test_launch_failure_when_server_never_ready(self) -> None:
        """A server that never accepts connections reports an error."""
        with (
            patch("ocr_tts.queue._launch_server", return_value=MagicMock()),
            patch("ocr_tts.queue._wait_for_server", return_value=False),
        ):
            result = execute_action(self._item())
        assert result == {
            "status": "error",
            "reason": "TTS API server did not become ready",
        }


class TestSequenceAction:
    """Tests for the sequence hotkey action."""

    def _item(self, *actions: HotkeyAction) -> HotkeyConfigItem:
        """Provide a sequence binding with the given sub-actions."""
        return HotkeyConfigItem(
            hotkey="<ctrl>+<shift>+r",
            action=HotkeyAction.SEQUENCE,
            actions=list(actions),
        )

    def test_sequence_runs_steps_in_order(self) -> None:
        """Sub-actions execute in declaration order."""
        calls: list[str] = []

        def fake_execute(item: HotkeyConfigItem) -> dict[str, Any]:
            calls.append(item.action.value)
            return {"status": "ok"}

        item = self._item(HotkeyAction.SHUTDOWN, HotkeyAction.LAUNCH)
        with patch(
            "ocr_tts.hotkey_watcher.execute_action", side_effect=fake_execute
        ) as exec_mock:
            result = execute_action(item)
        assert calls == ["shutdown", "launch"]
        assert result["status"] == "ok"
        assert [s["action"] for s in result["steps"]] == ["shutdown", "launch"]
        # Sub-steps receive copies bound to the step action.
        executed_actions = [call.args[0].action for call in exec_mock.call_args_list]
        assert executed_actions == [HotkeyAction.SHUTDOWN, HotkeyAction.LAUNCH]

    def test_sequence_stops_at_first_failure(self) -> None:
        """Execution halts when a sub-action does not report ok."""

        def fake_execute(_item: HotkeyConfigItem) -> dict[str, Any]:
            return {"status": "error", "reason": "nope"}

        item = self._item(HotkeyAction.SHUTDOWN, HotkeyAction.LAUNCH)
        with patch("ocr_tts.hotkey_watcher.execute_action", side_effect=fake_execute):
            result = execute_action(item)
        assert result["status"] == "error"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["reason"] == "nope"

    def test_sequence_empty_is_ok(self) -> None:
        """An empty action list succeeds without steps."""
        with patch("ocr_tts.hotkey_watcher.execute_action") as exec_mock:
            # Direct call to the private helper to avoid recursion via the
            # patched-out dispatcher.
            from ocr_tts.hotkey_watcher import _execute_sequence

            result = _execute_sequence(self._item())
        exec_mock.assert_not_called()
        assert result == {"status": "ok", "steps": []}


class TestSequenceConfigParsing:
    """Tests for parsing sequence bindings from YAML."""

    def test_yaml_round_trip(self, tmp_path: Any) -> None:
        """A sequence binding round-trips through load_config."""
        config_file = tmp_path / "seq.yaml"
        config_file.write_text(
            "hotkeys:\n"
            "  - hotkey: '<ctrl>+<shift>+r'\n"
            "    action: sequence\n"
            "    actions:\n"
            "      - shutdown\n"
            "      - launch\n"
        )
        config = load_config(config_file)
        assert len(config.hotkeys) == 1
        binding = config.hotkeys[0]
        assert binding.action is HotkeyAction.SEQUENCE
        assert binding.actions == [HotkeyAction.SHUTDOWN, HotkeyAction.LAUNCH]

    def test_invalid_sub_action_rejected(self, tmp_path: Any) -> None:
        """An unknown sub-action fails schema validation."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(
            "hotkeys:\n"
            "  - hotkey: '<ctrl>+<shift>+r'\n"
            "    action: sequence\n"
            "    actions: [bogus]\n"
        )
        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_example_yaml_has_restart_binding(self) -> None:
        """The bundled example ships a shutdown->launch restart binding."""
        config = load_config(get_default_config_path())
        sequences = [h for h in config.hotkeys if h.action is HotkeyAction.SEQUENCE]
        assert len(sequences) == 1
        assert sequences[0].actions == [
            HotkeyAction.SHUTDOWN,
            HotkeyAction.LAUNCH,
        ]
