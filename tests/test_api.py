"""Tests for the FastAPI streaming TTS server."""

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from piper import AudioChunk

from ocr_tts import api as api_module
from ocr_tts.api import app
from ocr_tts.player import AudioSink


@pytest.fixture
def client() -> TestClient:
    """Provide a FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_api_state() -> Iterator[None]:
    """Reset module-level server state between tests."""
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
    api_module._playback_sink = api_module.SounddeviceSink()
    api_module._synthesis_start_time = None
    api_module._enqueued_count = 0
    api_module._processed_count = 0
    api_module._first_audio_played_count = 0
    api_module._last_item_synthesis_s = None
    api_module._last_item_piper_latency_s = None
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
    api_module._playback_sink = api_module.SounddeviceSink()
    api_module._synthesis_start_time = None
    api_module._enqueued_count = 0
    api_module._processed_count = 0
    api_module._first_audio_played_count = 0
    api_module._last_item_synthesis_s = None
    api_module._last_item_piper_latency_s = None


async def _async_wait_until(
    predicate: Callable[[], bool], timeout: float = 5.0
) -> None:
    """Await asynchronously until a predicate becomes true or timeout."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise TimeoutError("Condition not met in time")
        await asyncio.sleep(0.01)


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


class TestSynthesizeEndpoint:
    """Tests for POST /synthesize."""

    @patch("ocr_tts.api.get_or_create_tts")
    def test_returns_wav(
        self,
        mock_get_tts: MagicMock,
        client: TestClient,
    ) -> None:
        """Test that POST /synthesize returns WAV audio."""
        tts = MagicMock()
        tts.synthesize.return_value = iter([make_chunk()])
        mock_get_tts.return_value = tts

        response = client.post(
            "/synthesize",
            json={"text": "Hello world"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        assert len(response.content) > 0

    @patch("ocr_tts.api.get_or_create_tts")
    def test_returns_400_on_no_audio(
        self,
        mock_get_tts: MagicMock,
        client: TestClient,
    ) -> None:
        """Test that POST /synthesize returns 400 for empty audio."""
        tts = MagicMock()
        tts.synthesize.return_value = iter([make_chunk()])
        # No chunks -> generate_wav returns b""
        mock_get_tts.return_value = tts

        # Force empty by making synthesize return nothing
        tts.synthesize.return_value = iter([])
        response = client.post(
            "/synthesize",
            json={"text": " "},
        )
        assert response.status_code == 400


class TestSynthesizeStreamEndpoint:
    """Tests for POST /synthesize/stream."""

    @patch("ocr_tts.api.get_or_create_tts")
    def test_streams_raw_pcm(
        self,
        mock_get_tts: MagicMock,
        client: TestClient,
    ) -> None:
        """Test that POST /synthesize/stream streams raw PCM."""
        tts = MagicMock()
        tts.sample_rate = 22050
        tts.synthesize.return_value = iter([make_chunk()])
        mock_get_tts.return_value = tts

        response = client.post(
            "/synthesize/stream",
            json={"text": "Hello world"},
        )
        assert response.status_code == 200
        assert "audio/L16" in response.headers["content-type"]
        assert len(response.content) > 0


class TestVoicesEndpoint:
    """Tests for GET /voices."""

    @patch("ocr_tts.api.list_downloaded_voices")
    @patch("ocr_tts.api.get_voice_dir")
    def test_lists_voices(
        self,
        mock_get_dir: MagicMock,
        mock_list: MagicMock,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        """Test that GET /voices lists downloaded voices."""
        mock_get_dir.return_value = tmp_path
        mock_list.return_value = ["en_US-hfc_male-medium"]
        response = client.get("/voices")
        assert response.status_code == 200
        assert "voices" in response.json()
        assert "en_US-hfc_male-medium" in response.json()["voices"]
        assert "default" in response.json()


class TestQueueEndpoint:
    """Tests for POST /queue."""

    @patch("ocr_tts.api._start_queue_processor")
    def test_queue_text(
        self, mock_start_processor: MagicMock, client: TestClient
    ) -> None:
        """Test that POST /queue accepts text."""
        response = client.post("/queue", json={"text": "Hi there"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        mock_start_processor.assert_called()

    @patch("ocr_tts.api._start_queue_processor")
    def test_queue_item_carries_voice_and_speed(
        self, _mock_start_processor: MagicMock, client: TestClient
    ) -> None:
        """Queued items store their own voice and speed."""
        response = client.post(
            "/queue",
            json={
                "text": "Hi there",
                "voice": "en_US-lessac-medium",
                "speed": 1.5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["queue_size"] == 1
        text_queue = api_module._text_queue
        assert text_queue is not None
        assert text_queue.get_nowait() == ("Hi there", "en_US-lessac-medium", 1.5)
        assert text_queue.empty()

    @patch("ocr_tts.api._start_queue_processor")
    def test_queue_resolves_voice_alias(
        self, _mock_start_processor: MagicMock, client: TestClient
    ) -> None:
        """Voice aliases are resolved before being stored in the queue."""
        response = client.post(
            "/queue",
            json={
                "text": "Hi there",
                "voice": "male",
                "speed": 1.0,
            },
        )
        assert response.status_code == 200
        text_queue = api_module._text_queue
        assert text_queue is not None
        assert text_queue.get_nowait() == (
            "Hi there",
            "en_US-hfc_male-medium",
            1.0,
        )
        assert text_queue.empty()

    def test_queue_default_does_not_wait(self, client: TestClient) -> None:
        """Without ``wait`` the response carries no latency info."""
        with patch("ocr_tts.api.get_or_create_tts") as mock_get_tts:
            tts = MagicMock()
            tts.sample_rate = 22050
            tts.synthesize.return_value = iter([make_chunk()])
            mock_get_tts.return_value = tts
            with patch("ocr_tts.api._start_playback"):
                response = client.post("/queue", json={"text": "hi"})
        data = response.json()
        assert data["status"] == "queued"
        assert "synthesis_ms" not in data
        assert "latency_ms" not in data

    def test_queue_wait_returns_latency_info(self) -> None:
        """A verbose ``wait`` request blocks until audio *starts* playing."""
        api_module._playback_sink = FakeSink()
        tts = MagicMock()
        tts.sample_rate = 22050
        tts.synthesize.return_value = iter([make_chunk()])
        with (
            patch("ocr_tts.api.get_or_create_tts", return_value=tts),
            TestClient(app) as client,
        ):
            response = client.post("/queue", json={"text": "hi", "wait": True})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        # The wait returns as soon as the first chunk is played, so the
        # piper-to-speech latency must be reported (FakeSink opens/writes).
        # Synthesis time is reported too (single chunk synthesizes quickly).
        assert data["synthesis_ms"] is not None
        assert data["synthesis_ms"] >= 0
        assert data["latency_ms"] is not None
        assert data["latency_ms"] >= 0
        # The first-audio event fired for this item.
        assert api_module._first_audio_played_count >= 1

    def test_queue_wait_unblocks_at_first_audio_not_end_of_synthesis(self) -> None:
        """The verbose wait returns when audio *starts*, not when it ends.

        Uses a synthesizer that yields several chunks with a delay between
        them so total synthesis takes ~1s.  The wait must return as soon as
        the *first* chunk is played (latency_ms is set then), well before the
        last chunk is produced — i.e. the server-side wait is far shorter
        than the full synthesis duration.
        """
        api_module._playback_sink = FakeSink()
        tts = MagicMock()
        tts.sample_rate = 22050

        def slow_chunks() -> Iterator[AudioChunk]:
            yield make_chunk()
            # Remaining chunks take ~1s to produce after the first.
            time.sleep(0.5)
            yield make_chunk()
            time.sleep(0.5)
            yield make_chunk()

        tts.synthesize.return_value = slow_chunks()
        with (
            patch("ocr_tts.api.get_or_create_tts", return_value=tts),
            TestClient(app) as client,
        ):
            start = time.monotonic()
            response = client.post("/queue", json={"text": "hi", "wait": True})
            elapsed = time.monotonic() - start
        assert response.status_code == 200
        data = response.json()
        # Latency (first chunk) is reported as soon as audio starts.
        assert data["latency_ms"] is not None
        assert data["latency_ms"] >= 0
        # The first-audio event fired for this item.
        assert api_module._first_audio_played_count >= 1
        # Total synthesis of all three chunks is ~1s, but the wait returns at
        # the first chunk, so the server-side wait must be well under that.
        assert elapsed < 0.9, (
            f"verbose wait took {elapsed:.3f}s, expected < 0.9s (first chunk)"
        )


class TestQueueClearEndpoint:
    """Tests for POST /queue/clear."""

    @patch("ocr_tts.api._start_queue_processor")
    def test_clear_drains_text_and_audio(
        self, _mock_start_processor: MagicMock, client: TestClient
    ) -> None:
        """Clearing the queue wipes pending text and audio."""
        client.post("/queue", json={"text": "first"})
        client.post("/queue", json={"text": "second"})
        audio_queue = api_module._audio_queue
        assert audio_queue is not None
        audio_queue.put_nowait(make_chunk())
        audio_queue.put_nowait(make_chunk())

        response = client.post("/queue/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        assert data["queue_size"] == 0
        text_queue = api_module._text_queue
        assert text_queue is not None
        assert text_queue.empty()
        assert audio_queue.empty()

    def test_clear_bumps_generation(self, client: TestClient) -> None:
        """Clearing increments the generation used to drop stale audio."""
        assert api_module._clear_generation == 0
        response = client.post("/queue/clear")
        assert response.status_code == 200
        assert response.json()["status"] == "cleared"
        assert api_module._clear_generation == 1

    def test_clear_cancels_queue_processor(self) -> None:
        """Clearing the queue stops the background queue processor task."""
        # Use a persistent client context so the server event loop (and the
        # processor task it owns) stays alive between requests, mirroring
        # production uvicorn behaviour.
        api_module._playback_sink = FakeSink()
        with TestClient(app) as client:
            tts = MagicMock()
            tts.sample_rate = 22050
            tts.synthesize.return_value = iter([make_chunk()])
            with patch("ocr_tts.api.get_or_create_tts", return_value=tts):
                client.post("/queue", json={"text": "hello"})

            task = api_module._queue_processor_task
            assert task is not None
            assert not task.done()

            response = client.post("/queue/clear")
            assert response.status_code == 200
            assert response.json()["status"] == "cleared"
            # The processor task is stopped and no longer referenced.
            assert api_module._queue_processor_task is None
            assert task.done()

    def test_clear_restarts_processor_on_next_queue(self) -> None:
        """After a clear, a new queue request restarts the processor."""
        api_module._playback_sink = FakeSink()
        with TestClient(app) as client:
            tts = MagicMock()
            tts.sample_rate = 22050
            tts.synthesize.return_value = iter([make_chunk()])
            with patch("ocr_tts.api.get_or_create_tts", return_value=tts):
                client.post("/queue", json={"text": "first"})
                first_task = api_module._queue_processor_task
                first_response = client.post("/queue/clear")
                assert first_response.status_code == 200
                assert api_module._queue_processor_task is None

                client.post("/queue", json={"text": "second"})
                second_task = api_module._queue_processor_task
                assert second_task is not None
                assert not second_task.done()
                assert second_task is not first_task

    def test_stop_queue_processor_cancels_running_task(self) -> None:
        """The stop helper cancels and clears a running processor task."""

        async def scenario() -> bool:
            task = asyncio.create_task(asyncio.sleep(100))
            api_module._queue_processor_task = task
            await api_module._stop_queue_processor()
            return task.done()

        done = asyncio.run(scenario())
        assert done
        assert api_module._queue_processor_task is None


class TestShutdown:
    """Tests for the /shutdown endpoint used by `ocr-tts api close`."""

    def test_shutdown_returns_status_and_bumps_generation(
        self, client: TestClient
    ) -> None:
        """POST /shutdown confirms teardown and bumps the generation."""
        assert api_module._clear_generation == 0
        response = client.post("/shutdown")
        assert response.status_code == 200
        assert response.json() == {"status": "shutting_down"}
        assert api_module._clear_generation == 1

    def test_shutdown_sets_playback_shutdown(self, client: TestClient) -> None:
        """Shutdown flags the playback thread to stop and close its sink."""
        assert not api_module._playback_shutdown.is_set()
        response = client.post("/shutdown")
        assert response.status_code == 200
        assert api_module._playback_shutdown.is_set()

    @patch("ocr_tts.api._request_server_exit")
    def test_shutdown_requests_server_exit(
        self, mock_request_exit: MagicMock, client: TestClient
    ) -> None:
        """Shutdown asks the running uvicorn server to exit gracefully."""
        response = client.post("/shutdown")
        assert response.status_code == 200
        mock_request_exit.assert_called_once()

    def test_request_server_exit_sets_flag(self) -> None:
        """_request_server_exit sets should_exit on the known server."""
        fake_server = MagicMock()
        app.state.uvicorn_server = fake_server
        try:
            api_module._request_server_exit()
            assert fake_server.should_exit is True
        finally:
            app.state.uvicorn_server = None

    def test_request_server_exit_noop_without_server(self) -> None:
        """_request_server_exit is a no-op when no uvicorn server is known."""
        app.state.uvicorn_server = None
        api_module._request_server_exit()  # should not raise


class TestPlayback:
    """Tests for server-side audio playback."""

    @patch("ocr_tts.api.get_or_create_tts")
    def test_playback_writes_to_sink(
        self, mock_get_tts: MagicMock, client: TestClient
    ) -> None:
        """Queued text results in audio chunks written to the sink."""
        fake_sink = FakeSink()
        api_module._playback_sink = fake_sink

        tts = MagicMock()
        tts.sample_rate = 22050
        tts.synthesize.return_value = iter([make_chunk()])
        mock_get_tts.return_value = tts

        response = client.post("/queue", json={"text": "Hello"})
        assert response.status_code == 200

        deadline = time.monotonic() + 5.0
        while fake_sink.total_bytes() == 0:
            if time.monotonic() > deadline:
                raise TimeoutError("Audio never reached the sink")
            time.sleep(0.05)

        assert fake_sink.total_bytes() > 0
        assert fake_sink.format == (22050, 1, 2)

    def test_log_first_audio_latency_logs_and_resets(self, caplog: Any) -> None:
        """The first-audio latency is logged and the clock is consumed."""
        api_module._synthesis_start_time = time.time() - 0.5
        with caplog.at_level(logging.INFO, logger="ocr_tts.api"):
            api_module._log_first_audio_latency()
        assert "Piper-to-speech latency" in caplog.text
        assert api_module._synthesis_start_time is None

    def test_log_first_audio_latency_noop_without_clock(self, caplog: Any) -> None:
        """Without a recorded synthesis start, the helper just notes playback."""
        api_module._synthesis_start_time = None
        with caplog.at_level(logging.INFO, logger="ocr_tts.api"):
            api_module._log_first_audio_latency()
        assert "first chunk written" in caplog.text
        assert api_module._synthesis_start_time is None

    @patch("ocr_tts.api.get_or_create_tts")
    def test_playback_logs_piper_to_speech_latency(
        self, mock_get_tts: MagicMock, client: TestClient, caplog: Any
    ) -> None:
        """Streaming playback reports the synthesis-to-speech latency."""
        fake_sink = FakeSink()
        api_module._playback_sink = fake_sink

        tts = MagicMock()
        tts.sample_rate = 22050
        tts.synthesize.return_value = iter([make_chunk()])
        mock_get_tts.return_value = tts

        with caplog.at_level(logging.INFO, logger="ocr_tts.api"):
            client.post("/queue", json={"text": "Hello"})
            deadline = time.monotonic() + 5.0
            while fake_sink.total_bytes() == 0:
                if time.monotonic() > deadline:
                    raise TimeoutError("Audio never reached the sink")
                time.sleep(0.05)
            # Wait for the playback thread to record (and reset) the latency.
            deadline = time.monotonic() + 5.0
            while api_module._synthesis_start_time is not None:
                if time.monotonic() > deadline:
                    raise TimeoutError("Latency was never logged")
                time.sleep(0.05)

        assert "Piper-to-speech latency" in caplog.text
        assert api_module._synthesis_start_time is None

    def test_clear_stops_playback(self, client: TestClient) -> None:
        """Clearing the queue signals the playback thread to stop."""
        assert not api_module._playback_stop.is_set()
        response = client.post("/queue/clear")
        assert response.status_code == 200
        assert api_module._playback_stop.is_set()

    @patch("ocr_tts.api.get_or_create_tts")
    def test_playback_uses_explicit_queue_reference(
        self, mock_get_tts: MagicMock, client: TestClient
    ) -> None:
        """Playback thread reads from the queue passed at startup."""
        fake_sink = FakeSink()
        api_module._playback_sink = fake_sink

        tts = MagicMock()
        tts.sample_rate = 22050
        tts.synthesize.return_value = iter([make_chunk()])
        mock_get_tts.return_value = tts

        # Start playback via /queue; the thread receives audio_queue via
        # _start_playback(loop, audio_queue) and uses that reference.
        client.post("/queue", json={"text": "Hello"})

        deadline = time.monotonic() + 5.0
        while fake_sink.total_bytes() == 0:
            if time.monotonic() > deadline:
                raise TimeoutError("Audio never reached the sink")
            time.sleep(0.05)

        assert fake_sink.total_bytes() > 0

    @patch("ocr_tts.api.get_or_create_tts")
    def test_sentinel_closes_sink_after_utterance(
        self, mock_get_tts: MagicMock, client: TestClient
    ) -> None:
        """End-of-utterance sentinel closes the sink after each utterance."""
        fake_sink = FakeSink()
        api_module._playback_sink = fake_sink

        tts = MagicMock()
        tts.sample_rate = 22050
        tts.synthesize.return_value = iter([make_chunk()])
        mock_get_tts.return_value = tts

        client.post("/queue", json={"text": "Hello"})

        # Wait for the utterance to finish: the queue processor delivers a
        # chunk then a sentinel, and the playback thread closes the sink
        # when it receives the sentinel.
        deadline = time.monotonic() + 5.0
        while not fake_sink.closed:
            if time.monotonic() > deadline:
                raise TimeoutError("Sink never closed after utterance")
            time.sleep(0.05)

        assert fake_sink.total_bytes() > 0


class TestGetOrCreateTTS:
    """Tests for the voice-keyed TTS engine cache."""

    @patch("ocr_tts.api.PiperTTS")
    def test_caches_an_instance_per_voice(self, mock_piper: MagicMock) -> None:
        """Distinct voices get distinct engines; repeats are reused."""
        instances: list[MagicMock] = []

        def factory(*_args: Any, **_kwargs: Any) -> MagicMock:
            instance = MagicMock()
            instances.append(instance)
            return instance

        mock_piper.side_effect = factory

        first = api_module.get_or_create_tts(voice="en_US-lessac-medium")
        second = api_module.get_or_create_tts(voice="fr_FR-siwis-medium")
        again = api_module.get_or_create_tts(voice="en_US-lessac-medium")

        assert first is again
        assert first is not second
        assert mock_piper.call_count == 2
        assert len(instances) == 2

    @patch("ocr_tts.api.PiperTTS")
    def test_alias_male_resolves(self, mock_piper: MagicMock) -> None:
        """The 'male' alias resolves to the default male voice."""
        api_module.get_or_create_tts(voice="male")
        mock_piper.assert_called_once_with(
            voice="en_US-hfc_male-medium", voice_dir=None
        )

    @patch("ocr_tts.api.PiperTTS")
    def test_alias_female_resolves(self, mock_piper: MagicMock) -> None:
        """The 'female' alias resolves to the female voice."""
        api_module.get_or_create_tts(voice="female")
        mock_piper.assert_called_once_with(
            voice="en_US-hfc_female-medium", voice_dir=None
        )

    @patch("ocr_tts.api.PiperTTS")
    def test_unknown_voice_unchanged(self, mock_piper: MagicMock) -> None:
        """Unknown voice names are passed through unchanged."""
        api_module.get_or_create_tts(voice="en_US-lessac-medium")
        mock_piper.assert_called_once_with(voice="en_US-lessac-medium", voice_dir=None)


class TestQueueProcessor:
    """Tests for the background queue processing loop."""

    @patch("ocr_tts.api.get_or_create_tts")
    def test_loop_uses_per_item_voice_and_speed(self, mock_get_tts: MagicMock) -> None:
        """Each queued item is synthesized with its own voice and speed."""
        calls: list[tuple[str, str, float]] = []

        def fake_synthesize(text: str, speed: float = 1.0) -> Iterator[AudioChunk]:
            voice = str(mock_get_tts.call_args.kwargs["voice"])
            calls.append((text, voice, speed))
            yield make_chunk()

        def fake_get_tts(*_args: Any, **_kwargs: Any) -> MagicMock:
            tts = MagicMock()
            tts.synthesize.side_effect = fake_synthesize
            return tts

        mock_get_tts.side_effect = fake_get_tts

        async def scenario() -> list[tuple[str, str, float]]:
            text_queue: asyncio.Queue[tuple[str, str, float]] = asyncio.Queue()
            audio_queue: queue.Queue[Any] = queue.Queue()
            api_module._text_queue = text_queue
            api_module._audio_queue = audio_queue
            task = asyncio.create_task(api_module._queue_processor_loop())
            await asyncio.sleep(0)
            await text_queue.put(("first", "en_US-lessac-medium", 1.5))
            await text_queue.put(("second", "fr_FR-siwis-medium", 0.8))
            await _async_wait_until(
                lambda: len(calls) == 2 and audio_queue.qsize() == 4
            )
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return calls

        result = asyncio.run(scenario())
        assert result == [
            ("first", "en_US-lessac-medium", 1.5),
            ("second", "fr_FR-siwis-medium", 0.8),
        ]

    def test_synthesize_item_stops_after_clear(self) -> None:
        """A generation change mid-synthesis discards remaining chunks."""
        api_module._clear_generation = 0

        async def scenario() -> int:
            audio_queue: queue.Queue[Any] = queue.Queue()

            def chunk_source() -> Iterator[AudioChunk]:
                yield make_chunk()
                api_module._clear_generation = 1
                yield make_chunk()

            tts = MagicMock()
            tts.synthesize.return_value = chunk_source()
            await asyncio.get_running_loop().run_in_executor(
                None,
                api_module._synthesize_item,
                tts,
                "text",
                1.0,
                0,
                audio_queue,
            )
            return audio_queue.qsize()

        size = asyncio.run(scenario())
        assert size == 1

    def test_queue_processor_puts_sentinel_after_synthesis(self) -> None:
        """Queue processor emits a sentinel after each utterance."""
        api_module._clear_generation = 0

        async def scenario() -> list[Any]:
            text_queue: asyncio.Queue[tuple[str, str, float]] = asyncio.Queue()
            audio_queue: queue.Queue[Any] = queue.Queue()
            api_module._text_queue = text_queue
            api_module._audio_queue = audio_queue

            tts = MagicMock()
            tts.sample_rate = 22050
            tts.synthesize.return_value = iter([make_chunk()])

            with patch("ocr_tts.api.get_or_create_tts", return_value=tts):
                task = asyncio.create_task(api_module._queue_processor_loop())
                await asyncio.sleep(0)
                await text_queue.put(("hello", "en_US-lessac-medium", 1.0))
                await _async_wait_until(lambda: audio_queue.qsize() == 2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            items: list[Any] = []
            while not audio_queue.empty():
                items.append(audio_queue.get_nowait())
            return items

        items = asyncio.run(scenario())
        assert len(items) == 2
        assert isinstance(items[0], AudioChunk)
        assert items[0].sample_rate == 22050
        assert items[1] is api_module._AUDIO_SENTINEL

    @patch("ocr_tts.api.get_or_create_tts")
    def test_concurrent_queue_requests_deduplicate_processor(
        self, mock_get_tts: MagicMock, client: TestClient
    ) -> None:
        """Concurrent /queue requests must not spawn multiple processors."""
        fake_sink = FakeSink()
        api_module._playback_sink = fake_sink

        tts = MagicMock()
        tts.sample_rate = 22050
        # Use a side_effect to count how many times synthesize is called
        # and track which texts are synthesized.
        synthesized_texts: list[str] = []

        def track_synthesize(*_args: Any, **_kwargs: Any) -> Iterator[AudioChunk]:
            text = _args[0]
            synthesized_texts.append(text)
            yield make_chunk()

        tts.synthesize.side_effect = track_synthesize
        mock_get_tts.return_value = tts

        async def scenario() -> list[Any]:
            # Fire a single request to start the processor, then immediately
            # queue the same text again to trigger the race condition where
            # two processors might pick up the same item.
            loop = asyncio.get_event_loop()
            # First request creates the queue processor
            response1 = await loop.run_in_executor(
                None, lambda: client.post("/queue", json={"text": "Hello world"})
            )
            # Small delay then second request - both items should be in queue
            await asyncio.sleep(0.01)
            response2 = await loop.run_in_executor(
                None, lambda: client.post("/queue", json={"text": "Hello world"})
            )
            return [response1, response2]

        responses = asyncio.run(scenario())
        assert all(r.status_code == 200 for r in responses)

        # Wait for audio to reach the sink.
        deadline = time.monotonic() + 5.0
        while fake_sink.total_bytes() == 0:
            if time.monotonic() > deadline:
                raise TimeoutError("Audio never reached the sink")
            time.sleep(0.05)

        # There must be exactly one synthesized utterance per queued item.
        assert fake_sink.total_bytes() > 0
        # The key assertion: with 2 queued items ("Hello world" twice) and a
        # single processor, synthesize should be called exactly 2 times.
        # Without the lock, concurrent requests could spawn multiple processors
        # that both consume from the same queue, leading to duplicate synthesis.
        assert len(synthesized_texts) == 2
        assert all(text == "Hello world" for text in synthesized_texts)
