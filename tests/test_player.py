"""Tests for the real-time streaming playback module."""

import sys
import threading
import time
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import typer
from piper import AudioChunk
from typer.testing import CliRunner

from ocr_tts.player import AudioSink, SounddeviceSink, StreamingPlayer, app

FRAMES = 800
BYTES_PER_FRAME = 2  # 16-bit PCM, mono


def make_chunk(amplitude: float = 0.0, frames: int = FRAMES) -> AudioChunk:
    """Build a minimal AudioChunk with a constant-amplitude tone."""
    arr = np.full(frames, amplitude, dtype=np.float32)
    return AudioChunk(
        sample_rate=22050,
        sample_width=2,
        sample_channels=1,
        audio_float_array=arr,
        phonemes=[],
        phoneme_ids=[],
    )


class FakeSink(AudioSink):
    """Recording sink that captures opens/writes/closes thread-safely."""

    def __init__(self) -> None:
        """Initialize recording sink with empty collections."""
        self.writes: list[bytes] = []
        self.format: tuple[int, int, int] | None = None
        self.closed = False
        self._lock = threading.Lock()

    def open(self, sample_rate: int, channels: int, sample_width: int) -> None:
        """Record the format used to open the sink."""
        self.format = (sample_rate, channels, sample_width)

    def write(self, pcm: bytes) -> None:
        """Record a PCM block."""
        with self._lock:
            self.writes.append(pcm)

    def close(self) -> None:
        """Mark the sink as closed."""
        self.closed = True

    def total_bytes(self) -> int:
        """Return total number of PCM bytes written."""
        with self._lock:
            return sum(len(w) for w in self.writes)


class FakeTTS:
    """Minimal Synthesizer yielding one chunk per text and recording calls."""

    def __init__(self) -> None:
        """Initialize the fake TTS with an empty call log."""
        self.calls: list[tuple[str, float]] = []

    def synthesize(self, text: str, speed: float = 1.0) -> Iterator[AudioChunk]:
        """Yield one chunk and record the call."""
        self.calls.append((text, speed))
        yield make_chunk(amplitude=0.1)


class TestStreamingPlayer:
    """Tests for StreamingPlayer behavior."""

    def test_say_queues_and_plays_in_order(self) -> None:
        """Multiple texts enqueued at arbitrary times play in FIFO order."""
        tts = FakeTTS()
        sink = FakeSink()
        player = StreamingPlayer(tts, sink)
        player.start()
        player.say("first")
        player.say("second")
        player.say("third")
        player.stop()

        assert tts.calls == [("first", 1.0), ("second", 1.0), ("third", 1.0)]
        assert sink.format == (22050, 1, 2)
        assert len(sink.writes) == 3
        assert sink.total_bytes() == 3 * FRAMES * BYTES_PER_FRAME
        assert sink.closed is True

    def test_playback_starts_before_synthesis_finishes(self) -> None:
        """First chunk plays before the rest of the text is synthesized."""
        release = threading.Event()
        synthesized: list[str] = []

        class BlockingTTS:
            def synthesize(
                self, text: str, _speed: float = 1.0
            ) -> Iterator[AudioChunk]:
                synthesized.append(text)
                yield make_chunk(amplitude=0.1)
                release.wait(timeout=2)
                yield make_chunk(amplitude=0.2)

        sink = FakeSink()
        player = StreamingPlayer(BlockingTTS(), sink)
        player.start()
        player.say("hello")

        deadline = time.monotonic() + 2
        while sink.total_bytes() == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        # The first chunk reached the sink without the second being produced.
        assert sink.total_bytes() == FRAMES * BYTES_PER_FRAME

        release.set()
        player.stop()
        assert sink.total_bytes() == 2 * FRAMES * BYTES_PER_FRAME

    def test_context_manager_starts_and_stops(self) -> None:
        """Using the player as a context manager closes the sink on exit."""
        tts = FakeTTS()
        sink = FakeSink()
        with StreamingPlayer(tts, sink) as player:
            player.say("ctx")
        assert sink.closed is True
        assert tts.calls == [("ctx", 1.0)]

    def test_pending_counts_queued_text(self) -> None:
        """say() reports the number of queued text items."""
        player = StreamingPlayer(FakeTTS(), FakeSink())
        assert player.say("a") == 1
        assert player.say("b") == 2
        assert player.pending() == 2

    def test_double_start_is_no_op(self) -> None:
        """Calling start() twice does not spawn extra work."""
        tts = FakeTTS()
        sink = FakeSink()
        player = StreamingPlayer(tts, sink)
        player.start()
        player.start()
        player.say("x")
        player.stop()
        assert tts.calls == [("x", 1.0)]


class TestSounddeviceSink:
    """Tests for the sounddevice-backed sink."""

    def test_open_write_close(self) -> None:
        """The sink drives a sounddevice OutputStream lifecycle."""
        fake_sd = MagicMock()
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            sink = SounddeviceSink(blocksize=512)
            sink.open(22050, 1, 2)
            fake_sd.OutputStream.assert_called_once_with(
                samplerate=22050,
                channels=1,
                blocksize=512,
                dtype="int16",
                latency="low",
            )
            stream = fake_sd.OutputStream.return_value
            stream.start.assert_called_once()
            test_bytes = b"pcm-data"
            sink.write(test_bytes)
            # stream.write should be called with a numpy int16 array
            expected_array = np.frombuffer(test_bytes, dtype=np.int16)
            stream.write.assert_called_once()
            actual_args = stream.write.call_args[0]
            assert len(actual_args) == 1
            np.testing.assert_array_equal(actual_args[0], expected_array)
            sink.close()
            stream.stop.assert_called_once()
            stream.close.assert_called_once()

    def test_write_before_open_raises(self) -> None:
        """Writing before opening the sink raises RuntimeError."""
        sink = SounddeviceSink()
        with pytest.raises(RuntimeError, match="not open"):
            sink.write(b"x")


class TestLiveCLI:
    """Tests for the live playback CLI."""

    def test_app_is_typer(self) -> None:
        """The live app is a Typer instance."""
        assert isinstance(app, typer.Typer)

    @patch("ocr_tts.player.StreamingPlayer")
    @patch("ocr_tts.player.SounddeviceSink")
    @patch("ocr_tts.player.PiperTTS")
    def test_live_reads_stdin(
        self,
        _mock_tts: MagicMock,
        _mock_sink: MagicMock,
        mock_player: MagicMock,
        runner: CliRunner,
    ) -> None:
        """The live command enqueues each stdin line."""
        mock_player.return_value.__enter__.return_value = mock_player.return_value
        result = runner.invoke(
            app,
            ["--speed", "1.0"],
            input="hello there\nsecond line\n",
        )
        assert result.exit_code == 0
        mock_player.return_value.say.assert_any_call("hello there", 1.0)
        mock_player.return_value.say.assert_any_call("second line", 1.0)
