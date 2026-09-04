"""Tests for the background hotkey watcher service."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

import ocr_tts.hotkey_watcher as hw
from ocr_tts.cli import app as cli_app
from ocr_tts.hotkey_watcher import (
    HotkeyAction,
    HotkeyConfig,
    HotkeyConfigItem,
    app,
    build_callbacks,
    create_default_config,
    execute_action,
    get_default_config_path,
    load_config,
    resolve_config_path,
)


def write_config(path: Path, *items: HotkeyConfigItem) -> Path:
    """Serialize hotkey items to a YAML config file."""
    payload = {"hotkeys": [item.model_dump(mode="json") for item in items]}
    path.write_text(yaml.dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def speak_item() -> HotkeyConfigItem:
    """Provide a speak-text binding."""
    return HotkeyConfigItem(hotkey="<ctrl>+<shift>+s", action=HotkeyAction.SPEAK_TEXT)


@pytest.fixture
def clear_item() -> HotkeyConfigItem:
    """Provide a queue-clear binding."""
    return HotkeyConfigItem(hotkey="<ctrl>+<shift>+x", action=HotkeyAction.QUEUE_CLEAR)


@pytest.fixture
def shutdown_item() -> HotkeyConfigItem:
    """Provide a shutdown binding."""
    return HotkeyConfigItem(hotkey="<ctrl>+<shift>+q", action=HotkeyAction.SHUTDOWN)


class TestHotkeyConfig:
    """Tests for configuration models and (de)serialization."""

    def test_default_config_has_four_bindings(self) -> None:
        """Default config contains the four built-in bindings."""
        config = create_default_config()
        assert len(config.hotkeys) == 4
        assert {item.action for item in config.hotkeys} == {
            HotkeyAction.SPEAK_TEXT,
            HotkeyAction.QUEUE_CLEAR,
            HotkeyAction.SHUTDOWN,
            HotkeyAction.SEND_REGION,
        }

    def test_speed_validation_rejects_out_of_range(self) -> None:
        """Speed outside 0.1-3.0 is rejected by validation."""
        with pytest.raises(ValueError, match="speed"):
            HotkeyConfigItem(
                hotkey="<ctrl>+s", action=HotkeyAction.SPEAK_TEXT, speed=5.0
            )

    def test_round_trip_yaml(
        self, tmp_path: Path, speak_item: HotkeyConfigItem
    ) -> None:
        """Config round-trips losslessly through YAML."""
        path = write_config(tmp_path / "hotkeys.yaml", speak_item)
        loaded = load_config(path)
        assert loaded == HotkeyConfig(hotkeys=[speak_item])

    def test_send_region_params_round_trip_yaml(self, tmp_path: Path) -> None:
        """send-region params (tesseract-cmd, save-image) survive YAML."""
        item = HotkeyConfigItem(
            hotkey="<ctrl>+<shift>+o",
            action=HotkeyAction.SEND_REGION,
            lang="deu",
            tesseract_cmd="tesseract",
            save_image="region_capture.png",
        )
        path = write_config(tmp_path / "hotkeys.yaml", item)
        assert load_config(path).hotkeys == [item]

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Loading a missing YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "missing.yaml")

    def test_load_empty_file_returns_empty_config(self, tmp_path: Path) -> None:
        """An empty YAML file yields an empty configuration."""
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_config(path).hotkeys == []

    def test_load_invalid_action_raises(self, tmp_path: Path) -> None:
        """Unknown action names are rejected at load time."""
        path = tmp_path / "bad.yaml"
        path.write_text("hotkeys:\n  - hotkey: '<ctrl>+z'\n    action: bogus\n")
        with pytest.raises(ValueError, match="action"):
            load_config(path)

    def test_load_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        """A YAML document that is not a mapping is rejected."""
        path = tmp_path / "list.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_config(path)


class TestConfigPathResolution:
    """Tests for default-path resolution and ~ expansion."""

    def test_default_config_path_points_at_example(self) -> None:
        """The default path is the project's hotkeys.example.yaml."""
        path = get_default_config_path()
        assert path.name == "hotkeys.example.yaml"
        assert path.is_file()

    def test_default_example_loads_with_pydantic_validation(self) -> None:
        """The bundled example YAML passes schema validation."""
        config = load_config(get_default_config_path())
        assert len(config.hotkeys) == 5
        assert {item.action for item in config.hotkeys} == {
            HotkeyAction.SPEAK_TEXT,
            HotkeyAction.QUEUE_CLEAR,
            HotkeyAction.SHUTDOWN,
            HotkeyAction.SEND_REGION,
            HotkeyAction.SEQUENCE,
        }
        assert all(item.hotkey for item in config.hotkeys)

    def test_resolve_expands_tilde_to_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A leading ~ is expanded to the user's home directory."""
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "definitely-not-here.yaml"
        target.write_text("", encoding="utf-8")
        assert resolve_config_path("~/definitely-not-here.yaml") == target

    def test_resolve_accepts_str_and_path(self, tmp_path: Path) -> None:
        """Both str and pathlib.Path inputs resolve identically."""
        path = write_config(tmp_path / "cfg.yaml")
        assert resolve_config_path(str(path)) == resolve_config_path(path)

    def test_resolve_missing_file_raises(self, tmp_path: Path) -> None:
        """A nonexistent file raises FileNotFoundError with the path."""
        missing = tmp_path / "nope.yaml"
        with pytest.raises(FileNotFoundError, match=r"nope\.yaml"):
            resolve_config_path(missing)

    def test_resolve_directory_raises(self, tmp_path: Path) -> None:
        """A directory is not a valid config file."""
        with pytest.raises(FileNotFoundError):
            resolve_config_path(tmp_path)


class TestExecuteAction:
    """Tests for hotkey action dispatch."""

    @patch("ocr_tts.hotkey_watcher.send_speak_request")
    def test_speak_text(
        self, mock_send: MagicMock, speak_item: HotkeyConfigItem
    ) -> None:
        """speak-text forwards text and settings to the queue client."""
        mock_send.return_value = {"queue_size": 3}
        result = execute_action(speak_item)
        mock_send.assert_called_once_with(
            speak_item.text,
            host="127.0.0.1",
            port=8000,
            voice=speak_item.voice,
            speed=1.0,
            verbose=False,
        )
        assert result == {"status": "ok", "queue_size": 3}

    @patch("ocr_tts.hotkey_watcher.send_clear_request")
    def test_queue_clear(
        self, mock_clear: MagicMock, clear_item: HotkeyConfigItem
    ) -> None:
        """queue-clear calls the clear endpoint with server settings."""
        mock_clear.return_value = {"queue_size": 0}
        result = execute_action(clear_item)
        mock_clear.assert_called_once_with(host="127.0.0.1", port=8000)
        assert result == {"status": "ok", "queue_size": 0}

    @patch("ocr_tts.hotkey_watcher.send_shutdown_request")
    def test_shutdown(
        self, mock_shutdown: MagicMock, shutdown_item: HotkeyConfigItem
    ) -> None:
        """Shutdown calls the shutdown endpoint with server settings."""
        mock_shutdown.return_value = {"status": "shutting_down"}
        result = execute_action(shutdown_item)
        mock_shutdown.assert_called_once_with(host="127.0.0.1", port=8000)
        assert result == {"status": "ok", "response": {"status": "shutting_down"}}

    @patch("ocr_tts.hotkey_watcher.run_ocr_region")
    def test_ocr_region_delegates(
        self, mock_run: MagicMock, speak_item: HotkeyConfigItem
    ) -> None:
        """ocr-region dispatches to the interactive OCR workflow."""
        item = speak_item.model_copy(update={"action": HotkeyAction.OCR_REGION})
        mock_run.return_value = {"status": "ok", "queue_size": 1}
        assert execute_action(item) == {"status": "ok", "queue_size": 1}
        mock_run.assert_called_once_with(item)

    @patch("ocr_tts.hotkey_watcher.run_send_region")
    def test_send_region_delegates(
        self, mock_run: MagicMock, speak_item: HotkeyConfigItem
    ) -> None:
        """send-region dispatches to the api send-region workflow."""
        item = speak_item.model_copy(update={"action": HotkeyAction.SEND_REGION})
        mock_run.return_value = {"status": "ok", "queue_size": 1}
        assert execute_action(item) == {"status": "ok", "queue_size": 1}
        mock_run.assert_called_once_with(item)

    def test_client_failure_is_caught(self, speak_item: HotkeyConfigItem) -> None:
        # A hard client failure raises typer.Exit; the watcher must not die.
        """Client exceptions become error results instead of crashes."""
        with patch.object(hw, "send_speak_request", side_effect=RuntimeError("boom")):
            result = execute_action(speak_item)
        assert result["status"] == "error"
        assert "boom" in result["reason"]

    @patch("ocr_tts.hotkey_watcher.send_speak_request")
    def test_custom_host_port_forwarded(
        self, mock_send: MagicMock, speak_item: HotkeyConfigItem
    ) -> None:
        """Custom host/port options reach the queue client."""
        item = speak_item.model_copy(update={"host": "localhost", "port": 9000})
        mock_send.return_value = {}
        execute_action(item)
        assert mock_send.call_args.kwargs["host"] == "localhost"
        assert mock_send.call_args.kwargs["port"] == 9000


class TestRunOcrRegion:
    """Tests for the interactive OCR region workflow."""

    def test_no_region_selected_is_skipped(self, speak_item: HotkeyConfigItem) -> None:
        """Empty region selection is reported as skipped."""
        with (
            patch.object(hw, "__name__"),
            patch(
                "ocr_tts.speak_region.select_region",
                return_value=MagicMock(width=0, height=0),
            ),
        ):
            result = hw.run_ocr_region(speak_item)
        assert result["status"] == "skipped"

    def test_blank_ocr_result_is_error(self, speak_item: HotkeyConfigItem) -> None:
        """OCR finding no text is reported as an error."""
        region = MagicMock(width=10, height=10)
        image = MagicMock()
        with (
            patch("ocr_tts.speak_region.select_region", return_value=region),
            patch("ocr_tts.speak_region.capture_selected_region", return_value=image),
            patch("ocr_tts.speak_region.image_is_blank", return_value=False),
            patch("ocr_tts.speak_region.extract_text", return_value=""),
        ):
            result = hw.run_ocr_region(speak_item)
        assert result == {"status": "error", "reason": "no text detected"}

    def test_successful_ocr_queues_speech(self, speak_item: HotkeyConfigItem) -> None:
        """Extracted text is queued for speech."""
        region = MagicMock(width=10, height=10)
        image = MagicMock()
        with (
            patch("ocr_tts.speak_region.select_region", return_value=region),
            patch("ocr_tts.speak_region.capture_selected_region", return_value=image),
            patch("ocr_tts.speak_region.image_is_blank", return_value=False),
            patch("ocr_tts.speak_region.extract_text", return_value="hello"),
            patch(
                "ocr_tts.speak_region.send_speak_request",
                return_value={"queue_size": 2},
            ) as mock_send,
        ):
            result = hw.run_ocr_region(speak_item)
        assert result == {"status": "ok", "queue_size": 2}
        assert mock_send.call_args.args[0] == "hello"

    def test_forwards_all_config_params(self, speak_item: HotkeyConfigItem) -> None:
        """All configurable params reach capture_and_queue_region (M12)."""
        item = speak_item.model_copy(
            update={
                "action": HotkeyAction.OCR_REGION,
                "voice": "female",
                "speed": 1.5,
                "host": "localhost",
                "port": 9000,
                "lang": "fra",
                "tesseract_cmd": "tesseract",
                "save_image": "region_capture.png",
            }
        )
        with (
            patch("ocr_tts.speak_region.OCRConfig") as mock_config_cls,
            patch(
                "ocr_tts.speak_region.select_region",
                return_value=MagicMock(width=10, height=10),
            ),
            patch(
                "ocr_tts.speak_region.capture_selected_region",
                return_value=MagicMock(),
            ),
            patch("ocr_tts.speak_region.image_is_blank", return_value=False),
            patch("ocr_tts.speak_region.extract_text", return_value="bonjour"),
            patch(
                "ocr_tts.speak_region.send_speak_request",
                return_value={"queue_size": 4},
            ) as mock_send,
        ):
            result = hw.run_ocr_region(item)
        assert result == {"status": "ok", "queue_size": 4}
        mock_config_cls.assert_called_once_with(
            lang="fra", tesseract_cmd="tesseract"
        )
        mock_send.assert_called_once()
        assert mock_send.call_args.args[0] == "bonjour"
        assert mock_send.call_args.kwargs["host"] == "localhost"
        assert mock_send.call_args.kwargs["port"] == 9000


class TestRunSendRegion:
    """Tests for the api send-region hotkey workflow."""

    def test_forwards_all_config_params(self, speak_item: HotkeyConfigItem) -> None:
        """All configurable params reach capture_and_queue_region."""
        item = speak_item.model_copy(
            update={
                "action": HotkeyAction.SEND_REGION,
                "voice": "female",
                "speed": 1.5,
                "host": "localhost",
                "port": 9000,
                "lang": "fra",
                "tesseract_cmd": "tesseract",
                "save_image": "region_capture.png",
            }
        )
        with (
            patch("ocr_tts.speak_region.OCRConfig") as mock_config_cls,
            patch(
                "ocr_tts.speak_region.select_region",
                return_value=MagicMock(width=10, height=10),
            ),
            patch(
                "ocr_tts.speak_region.capture_selected_region",
                return_value=MagicMock(),
            ),
            patch("ocr_tts.speak_region.image_is_blank", return_value=False),
            patch(
                "ocr_tts.speak_region.extract_text",
                return_value="bonjour",
            ) as mock_extract,
            patch(
                "ocr_tts.speak_region.send_speak_request",
                return_value={"queue_size": 4},
            ) as mock_send,
        ):
            result = hw.run_send_region(item)
        assert result == {"status": "ok", "queue_size": 4}
        assert mock_send.call_args.kwargs["voice"] == "female"
        assert mock_send.call_args.kwargs["speed"] == 1.5
        assert mock_send.call_args.kwargs["host"] == "localhost"
        assert mock_send.call_args.kwargs["port"] == 9000
        mock_config_cls.assert_called_once_with(
            lang="fra", tesseract_cmd="tesseract"
        )
        assert mock_extract.call_args.kwargs["config"] is mock_config_cls.return_value

    def test_save_image_param_saves_capture(self, speak_item: HotkeyConfigItem) -> None:
        """The save_image config value saves the captured region."""
        item = speak_item.model_copy(
            update={
                "action": HotkeyAction.SEND_REGION,
                "save_image": "region.png",
            }
        )
        image = MagicMock()
        with (
            patch(
                "ocr_tts.speak_region.select_region",
                return_value=MagicMock(width=10, height=10),
            ),
            patch("ocr_tts.speak_region.capture_selected_region", return_value=image),
            patch("ocr_tts.speak_region.image_is_blank", return_value=False),
            patch("ocr_tts.speak_region.extract_text", return_value="hi"),
            patch("ocr_tts.speak_region.send_speak_request", return_value={}),
        ):
            result = hw.run_send_region(item)
        image.save.assert_called_once_with("region.png")
        assert result["status"] == "ok"

    def test_no_region_selected_is_skipped(self, speak_item: HotkeyConfigItem) -> None:
        """Empty region selection is reported as skipped."""
        item = speak_item.model_copy(update={"action": HotkeyAction.SEND_REGION})
        with (
            patch(
                "ocr_tts.speak_region.select_region",
                return_value=MagicMock(width=0, height=0),
            ),
        ):
            result = hw.run_send_region(item)
        assert result["status"] == "skipped"

    def test_no_text_detected_is_error(self, speak_item: HotkeyConfigItem) -> None:
        """OCR finding no text is reported as an error."""
        item = speak_item.model_copy(update={"action": HotkeyAction.SEND_REGION})
        with (
            patch(
                "ocr_tts.speak_region.select_region",
                return_value=MagicMock(width=10, height=10),
            ),
            patch(
                "ocr_tts.speak_region.capture_selected_region",
                return_value=MagicMock(),
            ),
            patch("ocr_tts.speak_region.image_is_blank", return_value=False),
            patch("ocr_tts.speak_region.extract_text", return_value=""),
        ):
            result = hw.run_send_region(item)
        assert result == {"status": "error", "reason": "no text detected"}


class TestBuildCallbacks:
    """Tests for callback construction."""

    def test_callbacks_map_hotkeys_to_callables(
        self,
        speak_item: HotkeyConfigItem,
        clear_item: HotkeyConfigItem,
    ) -> None:
        """Each configured hotkey maps to a callable callback."""
        callbacks = build_callbacks(HotkeyConfig(hotkeys=[speak_item, clear_item]))
        assert set(callbacks) == {
            "<ctrl>+<shift>+s",
            "<ctrl>+<shift>+x",
        }
        assert all(callable(cb) for cb in callbacks.values())

    def test_callback_runs_action_in_daemon_thread(
        self, speak_item: HotkeyConfigItem
    ) -> None:
        """Callbacks execute their action on a daemon thread."""
        done = threading.Event()
        captured: dict[str, object] = {}

        def fake_execute(item: HotkeyConfigItem) -> dict[str, object]:
            captured["item"] = item
            captured["thread"] = threading.current_thread()
            done.set()
            return {}

        callbacks = build_callbacks(HotkeyConfig(hotkeys=[speak_item]))
        with patch.object(hw, "execute_action", side_effect=fake_execute):
            callbacks[speak_item.hotkey]()
            assert done.wait(timeout=5.0)
        assert captured["item"] is speak_item
        thread = captured["thread"]
        assert isinstance(thread, threading.Thread)
        assert thread.daemon is True

    @pytest.mark.parametrize("action", list(HotkeyAction))
    def test_dispatch_routing(self, action: HotkeyAction) -> None:
        """UI actions go to the single-worker executor, others to threads."""
        item = HotkeyConfigItem(hotkey="<ctrl>+z", action=action)
        with (
            patch.object(hw, "_ui_executor") as mock_executor,
            patch.object(hw, "_run_guarded") as mock_guarded,
        ):
            hw.dispatch_action(item)
            if action in (HotkeyAction.OCR_REGION, HotkeyAction.SEND_REGION):
                mock_executor.submit.assert_called_once_with(mock_guarded, item)
            else:
                mock_executor.submit.assert_not_called()

    def test_send_region_reuses_same_ui_thread(
        self, speak_item: HotkeyConfigItem
    ) -> None:
        """Consecutive region selections run on one shared UI worker."""
        item = speak_item.model_copy(update={"action": HotkeyAction.SEND_REGION})
        threads: list[threading.Thread] = []
        original_submit = hw._ui_executor.submit

        def capturing_submit(fn: object, *args: object) -> object:
            def wrapper() -> None:
                fn(*args)  # type: ignore[operator]
                threads.append(threading.current_thread())

            return original_submit(wrapper)

        with (
            patch.object(hw._ui_executor, "submit", side_effect=capturing_submit),
            patch.object(
                hw,
                "execute_action",
                side_effect=lambda _item: {"status": "ok"},
            ),
        ):
            callbacks = build_callbacks(HotkeyConfig(hotkeys=[item]))
            callbacks[item.hotkey]()
            callbacks[item.hotkey]()
            deadline = threading.Event()
            for _ in range(100):
                if len(threads) == 2:
                    break
                deadline.wait(0.05)
        assert len(threads) == 2
        assert threads[0] is threads[1]


class TestHotkeyWatcher:
    """Tests for the watcher lifecycle."""

    def test_stop_without_start_is_noop(self) -> None:
        """Stopping before start is a safe no-op."""
        watcher = hw.HotkeyWatcher(create_default_config())
        watcher.stop()
        assert watcher.running is False

    def test_running_false_before_start(self) -> None:
        """The watcher reports not-running before start."""
        watcher = hw.HotkeyWatcher(create_default_config())
        assert watcher.running is False


class TestHotkeyCLI:
    """Tests for the hotkey-watcher Typer commands."""

    def test_generate_config_command_removed(self, runner: CliRunner) -> None:
        """The generate-config subcommand no longer exists."""
        result = runner.invoke(app, ["generate-config"])
        assert result.exit_code != 0
        help_result = runner.invoke(app, ["--help"])
        assert "generate-config" not in help_result.output

    def test_start_missing_config_fails(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Starting with a missing config file exits non-zero."""
        result = runner.invoke(app, ["start", "--config", str(tmp_path / "no.yaml")])
        assert result.exit_code != 0

    def test_registered_under_main_cli(self) -> None:
        """The group is reachable as ``ocr-tts hotkey-watcher``."""
        result = CliRunner().invoke(cli_app, ["hotkey-watcher", "--help"])
        assert "start" in result.output
