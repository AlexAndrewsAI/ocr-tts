"""Additional coverage tests for the streaming player and live command."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from piper import AudioChunk
from typer.testing import CliRunner

from ocr_tts.player import AudioSink, StreamingPlayer, app


def make_chunk() -> AudioChunk:
    """Create a minimal AudioChunk for testing."""
    return AudioChunk(
        sample_rate=22050,
        sample_width=2,
        sample_channels=1,
        audio_float_array=np.zeros(160, dtype=np.float32),
        phonemes=[],
        phoneme_ids=[],
    )


class RecordingSink(AudioSink):
    """In-memory sink recording writes and optionally failing."""

    def __init__(self, fail_write: bool = False) -> None:
        """Initialize the recording sink.

        Args:
            fail_write: When True, every write raises a RuntimeError.

        """
        self.writes: list[bytes] = []
        self.closed = False
        self._fail_write = fail_write

    def open(self, sample_rate: int, channels: int, sample_width: int) -> None:
        """Record nothing; opening is a no-op."""

    def write(self, pcm: bytes) -> None:
        """Record or fail the PCM write depending on configuration."""
        if self._fail_write:
            raise RuntimeError("device underrun")
        self.writes.append(pcm)

    def close(self) -> None:
        """Mark the sink as closed."""
        self.closed = True


class TestStreamingPlayerEdgeCases:
    """Tests for StreamingPlayer error handling."""

    def test_stop_before_start_is_noop(self) -> None:
        """Stopping an un-started player returns immediately."""
        sink = RecordingSink()
        player = StreamingPlayer(MagicMock(), sink)
        player.stop()
        assert player._started is False
        assert not sink.closed

    def test_synthesis_error_is_recorded(self) -> None:
        """A synthesis failure is stored on the player."""

        def boom() -> Any:
            raise RuntimeError("model exploded")
            yield make_chunk()  # pragma: no cover - makes this a generator

        tts = MagicMock()
        tts.synthesize.return_value = boom()
        player = StreamingPlayer(tts, RecordingSink())
        with player:
            player.say("hello")
        assert isinstance(player._error, RuntimeError)
        assert "model exploded" in str(player._error)

    def test_playback_error_is_recorded(self) -> None:
        """A sink write failure is stored on the player."""
        tts = MagicMock()

        def chunks() -> Any:
            yield make_chunk()

        tts.synthesize.return_value = chunks()
        player = StreamingPlayer(tts, RecordingSink(fail_write=True))
        with player:
            player.say("hello")
        assert isinstance(player._error, RuntimeError)
        assert "underrun" in str(player._error)

    def test_stop_keeps_started_when_thread_alive(self) -> None:
        """If stop() can't join a thread, the player stays marked as running."""
        player = StreamingPlayer(MagicMock(), RecordingSink())
        player._started = True
        dead_beef = MagicMock()
        dead_beef.is_alive.return_value = True
        player._threads = [dead_beef]
        player.stop()
        # _started is still True, so the player isn't declared stopped
        assert player._started is True
        # The sink is not closed (which only happens when _started is set to False)
        assert not player.sink.closed  # type: ignore[attr-defined]
        # join was called with timeout
        dead_beef.join.assert_called_once()

    def test_start_refuses_when_old_thread_alive(self) -> None:
        """If start() sees a still-alive thread from a previous run, it refuses."""
        player = StreamingPlayer(MagicMock(), RecordingSink())
        alive_thread = MagicMock()
        alive_thread.is_alive.return_value = True
        player._threads = [alive_thread]
        player._started = False
        with patch("ocr_tts.player.threading.Thread") as mock_thread_factory:
            player.start()
        # start() joined the straggler and refused to start
        alive_thread.join.assert_called_once()
        # _started is still False (player never started)
        assert player._started is False
        # No new threads were spawned
        mock_thread_factory.assert_not_called()

    def test_start_clears_previous_error(self) -> None:
        """A previous error is cleared when a player is restarted."""
        player = StreamingPlayer(MagicMock(), RecordingSink())
        player._error = RuntimeError("old error")
        player.start()
        # _error is cleared at the start of start()
        assert player._error is None


class TestLiveCommand:
    """Tests for the live playback CLI command."""

    def test_engine_failure_exits_nonzero(self, runner: CliRunner) -> None:
        """An engine load failure reports the error and exits 1."""
        with patch("ocr_tts.player.PiperTTS", side_effect=RuntimeError("no model")):
            result = runner.invoke(app, [])
        assert result.exit_code == 1
        assert "Error in live playback" in result.output

    def test_lines_are_spoken(self, runner: CliRunner) -> None:
        """Non-empty stdin lines are passed to the player."""
        tts = MagicMock()
        tts.synthesize.return_value = iter([make_chunk()])
        sink = RecordingSink()
        with (
            patch("ocr_tts.player.PiperTTS", return_value=tts),
            patch("ocr_tts.player.SounddeviceSink", return_value=sink),
        ):
            result = runner.invoke(app, [], input="first line\n\nsecond line\n")
        assert result.exit_code == 0
