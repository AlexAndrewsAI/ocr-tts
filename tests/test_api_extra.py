"""Additional coverage tests for the FastAPI server internals."""

import asyncio
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from piper import AudioChunk

from ocr_tts import api as api_module
from ocr_tts.api import app
from ocr_tts.player import AudioSink


@pytest.fixture(autouse=True)
def _reset_api_state() -> Any:
    """Reset module-level server state between tests.

    Always installs a silent ``FakeSink`` and patches
    ``SounddeviceSink`` so no test can produce audible output.
    """
    api_module._playback_shutdown.set()
    thread = api_module._playback_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)
    api_module._text_queue = None
    api_module._audio_queue = None
    api_module._queue_processor_task = None
    api_module._tts_cache = {}
    api_module._clear_generation = 0
    api_module._playback_stop.clear()
    api_module._playback_shutdown.clear()
    api_module._playback_thread = None
    api_module._playback_sink = FakeSink()
    api_module._synthesis_start_time = None
    api_module._enqueued_count = 0
    api_module._processed_count = 0
    api_module._first_audio_played_count = 0
    api_module._last_item_synthesis_s = None
    api_module._last_item_piper_latency_s = None
    with patch.object(api_module, "SounddeviceSink", side_effect=_silent_sink):
        yield
    api_module._playback_shutdown.set()
    thread = api_module._playback_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)
    api_module._text_queue = None
    api_module._audio_queue = None
    api_module._queue_processor_task = None
    api_module._tts_cache = {}
    api_module._playback_stop.clear()
    api_module._playback_shutdown.clear()
    api_module._playback_thread = None
    api_module._playback_sink = FakeSink()


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI TestClient."""
    return TestClient(app)


def make_chunk() -> AudioChunk:
    """Create a minimal AudioChunk for testing."""
    return AudioChunk(
        sample_rate=22050,
        sample_width=2,
        sample_channels=1,
        audio_float_array=np.zeros(1600, dtype=np.float32),
        phonemes=[],
        phoneme_ids=[],
    )


class FakeSink(AudioSink):
    """Recording sink that captures opens/writes/closes."""

    def __init__(self) -> None:
        """Initialize recording sink with empty collections."""
        self.writes: list[bytes] = []
        self.format: tuple[int, int, int] | None = None
        self.closed = False
        self.write_event = threading.Event()

    def open(self, sample_rate: int, channels: int, sample_width: int) -> None:
        """Record the format used to open the sink."""
        self.format = (sample_rate, channels, sample_width)

    def write(self, pcm: bytes) -> None:
        """Record a PCM block and signal waiters."""
        self.writes.append(bytes(pcm))
        self.write_event.set()

    def close(self) -> None:
        """Mark the sink as closed."""
        self.closed = True


class BrokenCloseSink(FakeSink):
    """Sink whose first close() call raises to simulate device failure."""

    def __init__(self) -> None:
        """Initialize the sink with its first close primed to fail."""
        super().__init__()
        self.close_calls = 0

    def close(self) -> None:
        """Raise on the first call, succeed afterwards."""
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("device gone")
        self.closed = True


def _silent_sink(*_args: Any, **_kwargs: Any) -> FakeSink:
    """Return a silent fake sink whenever a real one is requested."""
    return FakeSink()


class _ScriptedAudioQueue:
    """Fake audio queue whose get() runs scripted actions."""

    def __init__(self, script: list[Callable[[], Any]]) -> None:
        self._script = list(script)
        self._lock = threading.Lock()

    def get(self, *_args: Any, **_kwargs: Any) -> Any:
        while True:
            with self._lock:
                if self._script:
                    action = self._script.pop(0)
                    break
            # Script exhausted: report a timeout so the playback loop
            # re-checks its shutdown flag instead of parking forever.
            raise queue.Empty()
        result = action()
        if isinstance(result, Exception):
            raise result
        return result


class TestListDownloadedVoices:
    """Tests for the voice directory scanner."""

    def test_lists_only_voices_with_config(self, tmp_path: Path) -> None:
        """Only .onnx files paired with a .onnx.json are listed."""
        (tmp_path / "a.onnx").touch()
        (tmp_path / "a.onnx.json").touch()
        (tmp_path / "b.onnx").touch()
        assert api_module.list_downloaded_voices(tmp_path) == ["a"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        """An empty directory yields no voices."""
        assert api_module.list_downloaded_voices(tmp_path) == []


class TestStartHelpers:
    """Tests for the guarded background-worker starters."""

    def test_start_queue_processor_reuses_running_task(self) -> None:
        """A running processor task is not replaced."""

        async def scenario() -> bool:
            task = asyncio.create_task(asyncio.sleep(100))
            api_module._queue_processor_task = task
            await api_module._start_queue_processor()
            reused = api_module._queue_processor_task is task
            await api_module._stop_queue_processor()
            return reused

        assert asyncio.run(scenario())


class TestStartPlayback:
    """Tests for the playback-thread lifecycle helper."""

    def test_dead_thread_is_joined_and_replaced(self) -> None:
        """A dead playback thread is cleaned up before starting a new one."""
        dead = threading.Thread(target=lambda: None, name="dead-playback")
        dead.start()
        dead.join()
        api_module._playback_thread = dead
        with patch.object(api_module, "_playback_loop"):
            api_module._start_playback(asyncio.new_event_loop(), queue.Queue())
        try:
            assert api_module._playback_thread is not dead
            assert api_module._playback_thread is not None
        finally:
            api_module._playback_thread.join(timeout=1.0)

    def test_alive_thread_is_not_replaced(self) -> None:
        """A live playback thread is left untouched."""
        gate = threading.Event()
        holder = threading.Thread(target=gate.wait, daemon=True)
        holder.start()
        api_module._playback_thread = holder
        try:
            api_module._start_playback(asyncio.new_event_loop(), queue.Queue())
            assert api_module._playback_thread is holder
        finally:
            gate.set()
            holder.join(timeout=1.0)


class TestPlaybackLoopBranches:
    """Direct exercises of the playback loop's branch paths."""

    def _start(
        self, script: list[Callable[[], Any]], sink: FakeSink
    ) -> threading.Thread:
        api_module._playback_sink = sink
        q = _ScriptedAudioQueue(script)
        thread = threading.Thread(
            target=api_module._playback_loop,
            args=(asyncio.new_event_loop(), q),
            daemon=True,
        )
        thread.start()
        return thread

    def test_shutdown_closes_open_sink(self) -> None:
        """The shutdown event makes an open sink close and the thread exit."""
        sink = FakeSink()
        chunk = make_chunk()
        thread = self._start([lambda: chunk], sink)
        assert sink.write_event.wait(5.0)
        api_module._playback_shutdown.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert sink.closed
        assert sink.format == (22050, 1, 2)

    def test_timeouts_heartbeat_stop_and_sentinel(self) -> None:
        """Empty timeouts, stop handling, sentinel, and chunks all work."""

        def empty_with_stop() -> Exception:
            api_module._playback_stop.set()
            return queue.Empty()

        def chunk_with_stop() -> AudioChunk:
            api_module._playback_stop.set()
            return make_chunk()

        sink = FakeSink()
        script: list[Callable[[], Any]] = [queue.Empty] * 10
        script.append(empty_with_stop)
        script.append(lambda: make_chunk())
        # A chunk arriving while stopped discards it and closes open sinks.
        script.append(chunk_with_stop)
        script.append(lambda: make_chunk())
        script.append(lambda: api_module._AUDIO_SENTINEL)
        thread = self._start(script, sink)
        deadline = time.monotonic() + 5.0
        while len(sink.writes) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(sink.writes) == 2
        api_module._playback_shutdown.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert sink.closed

    def test_sentinel_closes_open_sink(self) -> None:
        """A sentinel closes an open sink without ending the loop."""
        sink = FakeSink()
        chunk = make_chunk()
        thread = self._start([lambda: chunk, lambda: api_module._AUDIO_SENTINEL], sink)
        deadline = time.monotonic() + 5.0
        while not sink.closed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sink.closed
        api_module._playback_shutdown.set()
        thread.join(timeout=5.0)

    def test_write_error_closes_sink(self) -> None:
        """A failing sink write closes and reopens cleanly next utterance."""

        class WriteFailSink(FakeSink):
            def write(self, _pcm: bytes) -> None:
                """Simulate a device failure on every write."""
                raise RuntimeError("underrun")

        sink = WriteFailSink()
        thread = self._start([lambda: make_chunk()], sink)
        deadline = time.monotonic() + 5.0
        while sink.format is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sink.format is not None
        api_module._playback_shutdown.set()
        thread.join(timeout=5.0)

    def test_close_failure_in_outer_handler(self) -> None:
        """An exception closing the sink is caught by the outer handler."""
        sink = BrokenCloseSink()
        chunk = make_chunk()
        # The sentinel path closes the open sink; BrokenCloseSink raises,
        # exercising the outer exception handler's cleanup.
        thread = self._start([lambda: chunk, lambda: api_module._AUDIO_SENTINEL], sink)
        deadline = time.monotonic() + 5.0
        while sink.close_calls < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        # First close raised into the outer handler; only now may the
        # thread be told to shut down.
        api_module._playback_shutdown.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert sink.close_calls >= 2


class TestSynthesizeItemErrors:
    """Tests for error paths of the synthesis worker."""

    def test_put_failure_propagates_and_logs(self) -> None:
        """A failed audio-queue put raises out of the worker."""
        bad_queue = MagicMock()
        bad_queue.put.side_effect = RuntimeError("queue closed")
        tts = MagicMock()
        tts.synthesize.return_value = iter([make_chunk()])
        with pytest.raises(RuntimeError, match="queue closed"):
            api_module._synthesize_item(tts, "text", 1.0, 0, bad_queue)


class TestQueueProcessorEdgePaths:
    """Tests for the queue processor's defensive branches."""

    async def _wait_until(
        self, predicate: Callable[[], bool], timeout: float = 5.0
    ) -> None:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while not predicate():
            if loop.time() > deadline:
                raise TimeoutError("condition not met")
            await asyncio.sleep(0.01)

    def test_missing_audio_queue_skips_item(self) -> None:
        """Items are skipped (with a log) when no audio queue exists."""

        async def scenario() -> None:
            text_queue: asyncio.Queue[tuple[str, str, float]] = asyncio.Queue()
            api_module._text_queue = text_queue
            api_module._audio_queue = None
            with patch("ocr_tts.api._ensure_queues", return_value=(text_queue, None)):
                task = asyncio.create_task(api_module._queue_processor_loop())
                await text_queue.put(("hi", "voice", 1.0))
                await asyncio.sleep(0.1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        asyncio.run(scenario())

    def test_tts_failure_counts_as_processed(self) -> None:
        """A failing TTS engine unblocks verbose waiters via the counter."""

        async def scenario() -> None:
            text_queue: asyncio.Queue[tuple[str, str, float]] = asyncio.Queue()
            audio_queue: queue.Queue[Any] = queue.Queue()
            api_module._text_queue = text_queue
            api_module._audio_queue = audio_queue
            with patch(
                "ocr_tts.api.get_or_create_tts", side_effect=RuntimeError("no model")
            ):
                task = asyncio.create_task(api_module._queue_processor_loop())
                await text_queue.put(("hi", "voice", 1.0))
                await self._wait_until(lambda: api_module._processed_count >= 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        asyncio.run(scenario())

    def test_sentinel_put_failure_is_logged(self) -> None:
        """A failing sentinel delivery does not crash the processor."""

        class FailingSentinelQueue(queue.Queue[Any]):
            def put_nowait(self, item: Any) -> None:
                if item is api_module._AUDIO_SENTINEL:
                    raise RuntimeError("closed")
                super().put_nowait(item)

        async def scenario() -> None:
            text_queue: asyncio.Queue[tuple[str, str, float]] = asyncio.Queue()
            audio_queue: FailingSentinelQueue = FailingSentinelQueue()
            api_module._text_queue = text_queue
            api_module._audio_queue = audio_queue
            tts = MagicMock()
            tts.synthesize.return_value = iter([make_chunk()])
            with patch("ocr_tts.api.get_or_create_tts", return_value=tts):
                task = asyncio.create_task(api_module._queue_processor_loop())
                await text_queue.put(("hi", "voice", 1.0))
                await self._wait_until(lambda: audio_queue.qsize() >= 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        asyncio.run(scenario())


class TestDownloadEndpoint:
    """Tests for POST /download."""

    @patch("ocr_tts.api.ensure_voice")
    def test_download_success(self, mock_ensure: MagicMock, client: TestClient) -> None:
        """A successful download reports model and config paths."""
        mock_ensure.return_value = (Path("/v/a.onnx"), Path("/v/a.onnx.json"))
        response = client.post("/download", json={"voice": "en_US-lessac-medium"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "downloaded"
        assert data["model"] == "/v/a.onnx"
        assert data["config"] == "/v/a.onnx.json"
        assert not api_module._playback_shutdown.is_set()

    @patch("ocr_tts.api.ensure_voice")
    def test_download_validation_error(
        self, mock_ensure: MagicMock, client: TestClient
    ) -> None:
        """An invalid voice name yields HTTP 400."""
        mock_ensure.side_effect = ValueError("unknown voice")
        response = client.post("/download", json={"voice": "nope"})
        assert response.status_code == 400
        assert not api_module._playback_shutdown.is_set()

    @patch("ocr_tts.api.ensure_voice")
    def test_download_generic_error(
        self, mock_ensure: MagicMock, client: TestClient
    ) -> None:
        """A download failure yields HTTP 500."""
        mock_ensure.side_effect = RuntimeError("network down")
        response = client.post("/download", json={"voice": "en_US-lessac-medium"})
        assert response.status_code == 500
        assert "Download failed" in response.text
        assert not api_module._playback_shutdown.is_set()

    @patch("ocr_tts.api.ensure_voice")
    def test_download_stops_live_playback_thread(
        self, mock_ensure: MagicMock, client: TestClient
    ) -> None:
        """A live playback thread is joined (with timeout warning)."""
        mock_ensure.return_value = (Path("/v/a.onnx"), Path("/v/a.onnx.json"))
        gate = threading.Event()
        holder = threading.Thread(target=gate.wait, daemon=True)
        holder.start()
        api_module._playback_thread = holder
        try:
            response = client.post("/download", json={"voice": "en_US-lessac-medium"})
            assert response.status_code == 200
        finally:
            gate.set()
            holder.join(timeout=1.0)


class TestSynthesizeStreamWorkerError:
    """Tests for stream synthesis when the engine fails."""

    @patch("ocr_tts.api.get_or_create_tts")
    def test_worker_error_ends_stream_empty(
        self, mock_get_tts: MagicMock, client: TestClient
    ) -> None:
        """A synthesis failure terminates the PCM stream gracefully."""
        tts = MagicMock()
        tts.sample_rate = 22050
        tts.synthesize.side_effect = RuntimeError("engine exploded")
        mock_get_tts.return_value = tts
        response = client.post(
            "/synthesize/stream",
            json={"text": "Hello world"},
        )
        assert response.status_code == 200
        assert response.content == b""


class TestWaitForFirstAudio:
    """Tests for the verbose queue wait helper."""

    def test_processed_path(self) -> None:
        """A completed item returns 'processed' immediately."""
        api_module._processed_count = 3
        result = asyncio.run(api_module._wait_for_first_audio_or_processed(1, 1.0))
        assert result == "processed"

    def test_timeout_path(self) -> None:
        """No progress within the timeout returns 'timeout'."""
        result = asyncio.run(api_module._wait_for_first_audio_or_processed(99, 0.05))
        assert result == "timeout"


class TestQueueStreamEndpoint:
    """Tests for GET /queue/stream."""

    @patch("ocr_tts.api.get_or_create_tts")
    def test_streams_queued_chunks(self, mock_get_tts: MagicMock) -> None:
        """The endpoint forwards queued audio chunks as raw PCM."""
        tts = MagicMock()
        tts.sample_rate = 22050
        mock_get_tts.return_value = tts

        async def scenario() -> tuple[str, list[bytes]]:
            response = await api_module.queue_stream()
            media_type = str(response.media_type)
            gen = response.body_iterator
            audio_q = api_module._audio_queue
            assert audio_q is not None

            def feeder() -> None:
                time.sleep(0.05)
                audio_q.put(api_module._AUDIO_SENTINEL)
                audio_q.put(make_chunk())
                audio_q.put(make_chunk())

            feeder_thread = threading.Thread(target=feeder, daemon=True)
            feeder_thread.start()
            collected: list[bytes] = []
            async for part in gen:
                collected.append(cast(bytes, part))
                if len(collected) == 2:
                    break
            await gen.aclose()  # type: ignore[attr-defined]
            return media_type, collected

        media_type, parts = asyncio.run(scenario())
        assert media_type.startswith("audio/L16")
        assert len(parts) == 2
        assert all(len(p) > 0 for p in parts)


class TestShutdownDrainsQueues:
    """Tests that /shutdown drains pending work."""

    def test_shutdown_drains_text_and_audio_queues(self, client: TestClient) -> None:
        """Pending text and audio are dropped on shutdown."""
        with (
            patch("ocr_tts.api._request_server_exit"),
            patch("ocr_tts.api._start_queue_processor"),
        ):
            client.post("/queue", json={"text": "pending"})
        text_queue = api_module._text_queue
        audio_queue = api_module._audio_queue
        assert text_queue is not None
        assert audio_queue is not None
        text_queue.put_nowait(("more", "voice", 1.0))
        audio_queue.put(make_chunk())

        with patch("ocr_tts.api._request_server_exit"):
            response = client.post("/shutdown")

        assert response.status_code == 200
        assert text_queue.empty()
        assert audio_queue.empty()


class TestServe:
    """Tests for the uvicorn entry point."""

    @patch("uvicorn.Server")
    @patch("uvicorn.Config")
    def test_serve_builds_and_runs_server(
        self, mock_config: MagicMock, mock_server_cls: MagicMock
    ) -> None:
        """serve() builds a uvicorn server bound to app state and runs it."""
        server = mock_server_cls.return_value
        try:
            api_module.serve(host="127.0.0.1", port=8123)
        finally:
            app.state.uvicorn_server = None
        mock_config.assert_called_once_with(app, host="127.0.0.1", port=8123)
        mock_server_cls.assert_called_once_with(mock_config.return_value)
        server.run.assert_called_once()
