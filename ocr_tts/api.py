"""FastAPI server for streaming text-to-speech using Piper.

Provides HTTP endpoints for:

  - ``POST /synthesize`` — Generate complete WAV audio from text.
  - ``POST /synthesize/stream`` — Stream audio chunks as generated.
  - ``POST /queue`` — Add text to the synthesis queue (input streaming).
  - ``POST /queue/clear`` — Wipe the queue and stop playback immediately.
  - ``GET /queue/stream`` — Stream audio from queued text (output streaming).
  - ``POST /shutdown`` — Tear down the queue, playback, and subprocesses,
    then exit the server gracefully (used by ``ocr-tts api close``).
  - ``GET /voices`` — List downloaded voice models.
  - ``POST /download`` — Download a voice model.

Each ``POST /queue`` item carries its own ``voice`` and ``speed``.  The
queue processor synthesises each item with its own voice, so changing the
voice mid-queue affects only text queued from that point on; already
queued text keeps the settings it was submitted with.

All synthesis runs offline using ONNX Runtime via Piper.
"""

import asyncio
import contextlib
import io
import logging
import queue
import threading
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from ocr_tts import __version__
from ocr_tts.player import AudioSink, SounddeviceSink
from ocr_tts.text2speech import (
    DEFAULT_VOICE,
    DEFAULT_VOICE_DIR,
    PiperTTS,
    ensure_voice,
    get_voice_dir,
    resolve_voice_alias,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="OCR-TTS API",
    description="Streaming text-to-speech API using Piper",
    version=__version__,
)

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

# Cache of TTS engines keyed by (voice, voice_dir).  One engine is
# created per distinct voice so the queue can switch voices mid-play.
_tts_cache: dict[tuple[str, str], PiperTTS] = {}
_tts_lock = threading.Lock()

# Lock protecting _start_queue_processor and _start_playback against
# concurrent calls racing to create background workers.
_start_lock = threading.Lock()

# Queues for input queueing and output streaming
_text_queue: asyncio.Queue[tuple[str, str, float]] | None = None
_audio_queue: queue.Queue[Any] | None = None
_queue_processor_task: asyncio.Task | None = None

# Bumped on every /queue/clear.  In-flight synthesis that captured an
# older generation discards its remaining chunks, so playback stops.
_clear_generation = 0

# Sentinel signalling end of one text's audio output
_AUDIO_SENTINEL: object = object()

# Playback worker state — a daemon thread consumes _audio_queue and
# writes chunks to a SounddeviceSink so the server speaks aloud.
_playback_thread: threading.Thread | None = None
_playback_sink: AudioSink = SounddeviceSink()  # type: ignore[misc]
_playback_stop = threading.Event()
_playback_shutdown = threading.Event()

# Timestamp (from :func:`time.time`) recorded when the Piper model starts
# synthesizing the current utterance.  The playback thread reads it when it
# writes the first chunk so it can log the synthesis-to-speech latency.
# Guarded by the single-item serialization of the queue processor.
_synthesis_start_time: float | None = None

# Per-item accounting used to report latency to a verbose ``POST /queue``
# that asked the server to wait for the item to be synthesized.  Items are
# processed strictly serially by the queue processor, so the "last" item's
# counters correspond to the most recently completed utterance.
_enqueued_count = 0
_processed_count = 0
_last_item_synthesis_s: float | None = None
_last_item_piper_latency_s: float | None = None
# Number of items whose first audio chunk has actually been played.  This is
# incremented by the playback thread inside :func:`_log_first_audio_latency`
# the moment the first chunk of an utterance is written to the sink, so a
# verbose ``POST /queue`` can unblock as soon as the audio *starts* rather
# than waiting for the whole utterance to finish synthesizing.
_first_audio_played_count = 0

# How long a verbose ``POST /queue`` waits for its item to be processed
# before giving up and returning whatever latency data is available.
_QUEUE_WAIT_TIMEOUT_S = 60.0


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class SynthesizeRequest(BaseModel):
    """Request model for text synthesis."""

    text: str = Field(..., description="Text to synthesize")
    voice: str = Field(
        DEFAULT_VOICE,
        description="Piper voice name or alias (e.g., en_US-hfc_male-medium or male).",
    )
    speed: float = Field(
        1.0, ge=0.1, le=3.0, description="Speed multiplier (1.0 = normal)."
    )
    wait: bool = Field(
        default=False,
        description=(
            "Block until the item's audio has started playing and include "
            "latency info (synthesis_ms, latency_ms) in the response."
        ),
    )


class DownloadRequest(BaseModel):
    """Request model for voice download."""

    voice: str = Field(..., description="Voice name to download.")
    voice_dir: str | None = Field(
        default=None,
        description="Directory for voice files (default: .piper-voices).",
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def get_or_create_tts(
    voice: str = DEFAULT_VOICE, voice_dir: str | None = None
) -> PiperTTS:
    """Get or create a TTS engine for a voice (thread-safe).

    Engines are cached per ``(voice, voice_dir)`` pair, so different
    voices can be used by different queued items without reloading the
    same model twice.

    Args:
        voice: Piper voice name or friendly alias (e.g. ``male``).
        voice_dir: Directory for voice files.

    Returns:
        The cached PiperTTS instance for the requested voice.

    """
    resolved_voice = resolve_voice_alias(voice)
    key = (resolved_voice, voice_dir or DEFAULT_VOICE_DIR)
    with _tts_lock:
        tts = _tts_cache.get(key)
        if tts is None:
            logger.info("Creating new TTS engine for voice=%s", resolved_voice)
            tts = PiperTTS(voice=resolved_voice, voice_dir=voice_dir)
            _tts_cache[key] = tts
            logger.info("TTS engine created and cached for voice=%s", voice)
        else:
            logger.debug("Reusing cached TTS engine for voice=%s", voice)
    return tts


def list_downloaded_voices(voice_dir: Path) -> list[str]:
    """Return sorted list of downloaded voice names in a directory.

    Args:
        voice_dir: Directory to search for voice files.

    Returns:
        Sorted list of voice names (e.g., ``en_US-hfc_male-medium``).

    """
    voices: list[str] = []
    for onnx_file in voice_dir.glob("*.onnx"):
        voice_name = onnx_file.name.replace(".onnx", "")
        if (voice_dir / f"{voice_name}.onnx.json").exists():
            voices.append(voice_name)
    return sorted(voices)


async def _ensure_queues() -> tuple[asyncio.Queue, queue.Queue[Any]]:
    """Initialize queues if not already created.

    Returns:
        Tuple of (text_queue, audio_queue).

    """
    global _text_queue, _audio_queue
    if _text_queue is None:
        _text_queue = asyncio.Queue()
        logger.info("Created text_queue")
    else:
        logger.debug("text_queue already exists")
    if _audio_queue is None:
        _audio_queue = queue.Queue()
        logger.info("Created audio_queue")
    else:
        logger.debug("audio_queue already exists")
    return _text_queue, _audio_queue


async def _start_queue_processor() -> None:
    """Start the background queue processor if not already running."""
    global _queue_processor_task
    # The lock makes the check-then-create atomic across concurrent requests.
    with _start_lock:
        if _queue_processor_task is None or _queue_processor_task.done():
            logger.info("Starting queue processor task")
            _queue_processor_task = asyncio.create_task(_queue_processor_loop())
        else:
            logger.debug("Queue processor task already running")


async def _stop_queue_processor() -> None:
    """Cancel and join the background queue processor task, if any.

    Used by ``/queue/clear`` to stop processing so no pending or future
    items are synthesized until new text is queued again (which restarts
    the task via :func:`_start_queue_processor`).
    """
    global _queue_processor_task
    task = _queue_processor_task
    if task is not None and not task.done():
        logger.info("Cancelling queue processor task")
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        logger.info("Queue processor task stopped")
    _queue_processor_task = None


def _start_playback(
    loop: asyncio.AbstractEventLoop, audio_queue: queue.Queue[Any]
) -> None:
    """Start the background playback thread if not already running.

    Args:
        loop: The asyncio event loop that owns ``audio_queue``.
        audio_queue: Thread-safe queue providing audio chunks.

    """
    global _playback_thread
    # The lock makes the check-then-create atomic across concurrent requests.
    with _start_lock:
        if _playback_thread is not None and _playback_thread.is_alive():
            logger.debug("Playback thread already running")
            return
        if _playback_thread is not None:
            logger.warning(
                "Previous playback thread is dead, cleaning up before starting new one"
            )
            _playback_thread.join(timeout=1.0)
        logger.info(
            "Starting playback thread with loop=%s, audio_queue=%s",
            loop,
            id(audio_queue),
        )
        _playback_thread = threading.Thread(
            target=_playback_loop,
            args=(loop, audio_queue),
            name="tts-playback",
            daemon=True,
        )
        _playback_thread.start()
        logger.info("Playback thread started successfully")


def _log_first_audio_latency() -> None:
    """Log the time from Piper model start to the first played chunk.

    Reads :data:`_synthesis_start_time` (recorded by the synthesis worker
    when the model began processing the current utterance) and reports how
    long it took until the first audio chunk was actually written to the
    sink — i.e. when the server started speaking.  The measured latency is
    also stashed on :data:`_last_item_piper_latency_s` so a verbose
    ``POST /queue`` can return it to the client.

    Increments :data:`_first_audio_played_count` so a waiting verbose
    request can unblock the moment the audio *starts* playing, instead of
    having to wait for the entire utterance to finish synthesizing.
    """
    global _synthesis_start_time, _last_item_piper_latency_s, _first_audio_played_count
    if _synthesis_start_time is None:
        logger.info("Audio playback started (first chunk written)")
        return
    latency = time.time() - _synthesis_start_time
    _last_item_piper_latency_s = latency
    _first_audio_played_count += 1
    _synthesis_start_time = None
    logger.info(
        "Audio playback started — Piper-to-speech latency: %.3fs",
        latency,
    )
    _synthesis_start_time = None


def _playback_loop(
    loop: asyncio.AbstractEventLoop, audio_queue: queue.Queue[Any]
) -> None:
    """Background thread that plays audio chunks from the queue.

    Consumes AudioChunk objects from ``audio_queue`` and writes them
    to ``_playback_sink``.  The sink is opened on the first chunk and
    closed when ``_playback_stop`` is set (e.g. by /queue/clear) or
    when an end-of-utterance sentinel is received.

    Args:
        loop: Event loop that owns ``audio_queue``.
        audio_queue: Thread-safe queue providing audio chunks.

    """
    sink = _playback_sink
    opened = False
    iteration = 0
    consecutive_timeouts = 0
    logger.info(
        "Playback thread started with loop=%s, audio_queue=%s",
        loop,
        id(audio_queue),
    )
    while True:
        iteration += 1
        try:
            if _playback_shutdown.is_set():
                if opened:
                    logger.info("Playback thread shutting down, closing sink")
                    sink.close()
                    opened = False
                return
            try:
                logger.debug(
                    "Playback thread [%d]: waiting for chunk from audio_queue "
                    "(queue_id=%s)",
                    iteration,
                    id(audio_queue),
                )
                chunk = audio_queue.get(timeout=5.0)
                logger.debug(
                    "Playback thread [%d]: got chunk from audio_queue (queue_id=%s)",
                    iteration,
                    id(audio_queue),
                )
                consecutive_timeouts = 0  # Reset timeout counter on success
            except queue.Empty:
                consecutive_timeouts += 1
                logger.debug(
                    "Playback thread [%d]: timeout waiting for chunk "
                    "(queue_id=%s, consecutive_timeouts=%d)",
                    iteration,
                    id(audio_queue),
                    consecutive_timeouts,
                )
                if _playback_stop.is_set():
                    if opened:
                        logger.info("Playback stopped via clear_queue, closing sink")
                        sink.close()
                        opened = False
                    _playback_stop.clear()
                if consecutive_timeouts % 10 == 0:
                    logger.info(
                        "Playback thread [%d]: heartbeat - still waiting for "
                        "chunks (queue_id=%s)",
                        iteration,
                        id(audio_queue),
                    )
                continue
            except Exception as e:
                logger.error(
                    "Playback thread [%d]: error getting chunk from audio_queue: %s",
                    iteration,
                    e,
                )
                time.sleep(0.1)
                continue
            if chunk is _AUDIO_SENTINEL:
                logger.info(
                    "Playback thread [%d]: got sentinel, closing sink",
                    iteration,
                )
                if opened:
                    sink.close()
                    opened = False
                continue
            if _playback_stop.is_set():
                if opened:
                    logger.info("Playback stopped via clear_queue, closing sink")
                    sink.close()
                    opened = False
                _playback_stop.clear()
                continue
            if not opened:
                logger.info("Opening audio sink with chunk parameters")
                sink.open(
                    chunk.sample_rate,
                    chunk.sample_channels,
                    chunk.sample_width,
                )
                opened = True
                logger.info(
                    "Opened audio sink: sample_rate=%d, channels=%d, sample_width=%d",
                    chunk.sample_rate,
                    chunk.sample_channels,
                    chunk.sample_width,
                )
                _log_first_audio_latency()
            try:
                sink.write(chunk.audio_int16_bytes)
                logger.info(
                    "[%s] Playing audio chunk: %d bytes (sample_rate=%d, channels=%d)",
                    time.strftime("%H:%M:%S"),
                    len(chunk.audio_int16_bytes),
                    chunk.sample_rate,
                    chunk.sample_channels,
                )
            except Exception as e:
                logger.error("Playback thread: error writing to sink: %s", e)
                if opened:
                    sink.close()
                    opened = False
        except Exception as e:
            logger.exception(
                "Playback thread [%d]: unhandled exception in main loop: %s",
                iteration,
                e,
            )
            if opened:
                with contextlib.suppress(Exception):
                    sink.close()
                opened = False
            time.sleep(0.5)


def _synthesize_item(
    tts: PiperTTS,
    text: str,
    speed: float,
    generation: int,
    audio_queue: queue.Queue[Any],
) -> None:
    """Synthesize one queued item, forwarding chunks to the audio queue.

    Runs in a worker thread.  Chunks produced after the queue has been
    cleared (i.e. ``_clear_generation`` no longer equals ``generation``)
    are dropped, so clearing the queue stops playback of later items
    without leaving stale audio behind.

    Args:
        tts: TTS engine to use for this item.
        text: Text to synthesize.
        speed: Speed multiplier for this item.
        generation: Queue generation captured when the item was dequeued.
        audio_queue: Thread-safe queue receiving the audio chunks.

    """
    global _synthesis_start_time, _last_item_synthesis_s
    start_time = time.time()
    logger.info(
        "_synthesize_item called for: %s (generation=%d, audio_queue=%s)",
        text[:50],
        generation,
        id(audio_queue),
    )
    try:
        chunk_count = 0
        # Record the moment the Piper model begins processing so the playback
        # thread can report how long until the first audio is actually spoken.
        _synthesis_start_time = time.time()
        logger.info(
            "Piper model started processing text: %s (latency clock started at %.6f)",
            text[:50],
            _synthesis_start_time,
        )
        for chunk in tts.synthesize(text, speed):
            if generation != _clear_generation:
                elapsed = time.time() - start_time
                logger.info(
                    "Synthesizer generation changed (%d != %d), "
                    "discarding remaining chunks after %.3fs for: %s",
                    generation,
                    _clear_generation,
                    elapsed,
                    text[:50],
                )
                return
            chunk_count += 1
            try:
                audio_queue.put(chunk, timeout=5.0)
                logger.info(
                    "Chunk %d queued to audio_queue (sample_rate=%d, bytes=%d, "
                    "queue_id=%s)",
                    chunk_count,
                    chunk.sample_rate,
                    len(chunk.audio_int16_bytes),
                    id(audio_queue),
                )
            except Exception as e:
                logger.error(
                    "Failed to put chunk %d into audio_queue: %s",
                    chunk_count,
                    e,
                )
                raise
        elapsed = time.time() - start_time
        _last_item_synthesis_s = elapsed
        logger.info(
            "Synthesized %d chunks and queued audio for: %s (%.3fs)",
            chunk_count,
            text[:50],
            elapsed,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        logger.exception(
            "Synthesizer error after %.3fs for %s: %s",
            elapsed,
            text[:50],
            e,
        )
        raise


async def _queue_processor_loop() -> None:
    """Background task that processes queued text into audio chunks.

    Consumes text from the input queue, synthesizes speech with Piper
    using each item's own voice and speed, and forwards each audio chunk
    to the audio output queue for streaming.
    """
    global _processed_count
    text_queue, _ = await _ensure_queues()
    loop = asyncio.get_event_loop()
    logger.info("Queue processor loop started with loop=%s", loop)

    while True:
        logger.info(
            "Queue processor waiting for next item (queue_size=%d)",
            text_queue.qsize(),
        )
        text, voice, speed = await text_queue.get()
        logger.info(
            "Queue processor got item: text=%r, voice=%s, speed=%.1f",
            text[:80],
            voice,
            speed,
        )
        try:
            audio_queue = _audio_queue
            if audio_queue is None:
                logger.error(
                    "audio_queue is None, cannot synthesize item: %s", text[:80]
                )
                continue
            logger.info("Getting/creating TTS engine for voice=%s", voice)
            tts = get_or_create_tts(voice=voice)
            logger.info("TTS engine ready, starting synthesis for: %s", text[:80])
            await loop.run_in_executor(
                None,
                _synthesize_item,
                tts,
                text,
                speed,
                _clear_generation,
                audio_queue,
            )
            logger.info("Synthesis completed for: %s", text[:80])
            _processed_count += 1
        except Exception as exc:
            logger.exception("Error processing queued item %r: %s", text, exc)
            # Unblock any verbose waiters even though the item failed.
            _processed_count += 1
        finally:
            # Signal end of utterance so playback closes the sink and
            # waits for the next item instead of blocking forever.
            current_queue = _audio_queue
            if current_queue is not None:
                try:
                    current_queue.put_nowait(_AUDIO_SENTINEL)
                except Exception as exc:
                    logger.error("Failed to put sentinel into audio_queue: %s", exc)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/voices")
async def get_voices() -> dict[str, Any]:
    """List downloaded voice models.

    Returns:
        Dictionary with ``voices`` list and ``default`` voice name.

    """
    voice_dir = get_voice_dir()
    voices = list_downloaded_voices(voice_dir)
    return {"voices": voices, "default": DEFAULT_VOICE}


@app.post("/download")
async def download_voice(request: DownloadRequest) -> dict[str, Any]:
    """Download a voice model and config.

    Args:
        request: Contains ``voice`` name and optional ``voice_dir``.

    Returns:
        Dictionary with download status and file paths.

    """
    global _playback_sink

    # Log before any state changes
    logger.info(
        "Starting voice download: %s (previous voices will be invalidated)",
        request.voice,
    )

    # Stop playback before clearing state
    _playback_shutdown.set()
    if _playback_thread is not None and _playback_thread.is_alive():
        logger.info("Stopping playback thread for voice download")
        _playback_thread.join(timeout=2.0)
        if _playback_thread.is_alive():
            logger.warning("Playback thread did not stop cleanly")

    logger.info("Closing current audio sink")
    _playback_sink.close()

    # Clear the TTS cache to ensure newly downloaded voices are used
    with _tts_lock:
        old_cache_size = len(_tts_cache)
        _tts_cache.clear()
        logger.info(
            "TTS cache cleared (%d old voices removed) for new voice download: %s",
            old_cache_size,
            request.voice,
        )

    resolved_voice = resolve_voice_alias(request.voice)
    voice_dir = get_voice_dir(request.voice_dir)
    try:
        logger.info("Ensuring voice files exist for: %s", resolved_voice)
        model_path, config_path = ensure_voice(resolved_voice, voice_dir)
        logger.info("Voice files verified for: %s", resolved_voice)
    except ValueError as e:
        logger.error("Voice validation failed for %s: %s", request.voice, e)
        _playback_shutdown.clear()
        _playback_sink = SounddeviceSink()  # type: ignore[misc]
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        logger.error("Voice download failed for %s: %s", request.voice, e)
        _playback_shutdown.clear()
        _playback_sink = SounddeviceSink()  # type: ignore[misc]
        raise HTTPException(500, f"Download failed: {e}") from e

    logger.info(
        "Voice downloaded: %s (model: %s, config: %s)",
        resolved_voice,
        model_path,
        config_path,
    )

    # Allow playback to restart with new voice
    logger.info("Preparing for playback with new voice")
    _playback_shutdown.clear()

    return {
        "status": "downloaded",
        "voice": resolved_voice,
        "model": str(model_path),
        "config": str(config_path),
    }


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest) -> StreamingResponse:
    """Synthesize text to a complete WAV file (non-streaming).

    The entire audio is generated and returned as a WAV file.

    Args:
        request: Text, voice, and speed parameters.

    Returns:
        StreamingResponse containing WAV audio bytes.

    """
    logger.info(
        "Synthesizing complete WAV: %s (voice=%s, speed=%.1f)",
        request.text[:50],
        request.voice,
        request.speed,
    )
    tts = get_or_create_tts(voice=request.voice)
    loop = asyncio.get_event_loop()

    def _generate_wav() -> bytes:
        chunks = list(tts.synthesize(request.text, speed=request.speed))
        if not chunks:
            return b""
        first = chunks[0]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(first.sample_channels)
            wav.setsampwidth(first.sample_width)
            wav.setframerate(first.sample_rate)
            for chunk in chunks:
                wav.writeframes(chunk.audio_int16_bytes)
        return buf.getvalue()

    wav_bytes = await loop.run_in_executor(None, _generate_wav)
    if not wav_bytes:
        raise HTTPException(400, "No audio generated from text")

    logger.info(
        "Complete WAV generated: %s (%.3f KB)",
        request.text[:50],
        len(wav_bytes) / 1024,
    )

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
    )


@app.post("/synthesize/stream")
async def synthesize_stream(request: SynthesizeRequest) -> StreamingResponse:
    """Stream raw PCM audio chunks as they are generated.

    Output streaming — audio begins playing before all text has been
    processed.  Each chunk corresponds to one sentence of audio.

    Args:
        request: Text, voice, and speed parameters.

    Returns:
        StreamingResponse with raw PCM audio data.

    """
    tts = get_or_create_tts(voice=request.voice)
    sample_rate = tts.sample_rate
    loop = asyncio.get_event_loop()

    async def _generate() -> AsyncIterator[bytes]:
        chunk_queue: asyncio.Queue = asyncio.Queue()
        sentinel: object = object()

        def _worker() -> None:
            """Run Piper synthesis in a thread, forwarding to asyncio."""
            try:
                for chunk in tts.synthesize(request.text, speed=request.speed):
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(chunk), loop
                    ).result()
            except Exception as exc:
                logger.error("Synthesis error: %s", exc)
            finally:
                asyncio.run_coroutine_threadsafe(
                    chunk_queue.put(sentinel), loop
                ).result()

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            chunk = await chunk_queue.get()
            if chunk is sentinel:
                break
            yield chunk.audio_int16_bytes

    return StreamingResponse(
        _generate(),
        media_type=f"audio/L16; rate={sample_rate}; channels=1",
    )


async def _wait_for_first_audio_or_processed(index: int, timeout: float) -> str:
    """Wait until the *index*-th item either starts playing or finishes.

    A verbose ``POST /queue`` should return (and the client should compute
    its turnaround) the moment the audio *starts* — i.e. when the playback
    thread writes the first chunk of this utterance — not when the whole
    utterance has finished synthesizing, since the audio begins streaming
    several seconds before the last chunk is produced.

    The first-chunk event is signalled by
    :data:`_first_audio_played_count` (incremented in
    :func:`_log_first_audio_latency` when the first chunk is written).  On a
    headless server with no audio device the sink never opens, so the
    first-chunk event never fires; in that case we fall back to
    :data:`_processed_count` (end of synthesis) so the request still
    unblocks and can report ``latency_ms: n/a``.

    Args:
        index: The 1-based enqueue index to wait for.
        timeout: Maximum seconds to wait.

    Returns:
        ``"first_audio"`` if the first chunk played before the timeout,
        ``"processed"`` if synthesis completed first (or failed), and
        ``"timeout"`` if neither happened before the timeout elapsed.

    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if _first_audio_played_count >= index:
            return "first_audio"
        if _processed_count >= index:
            return "processed"
        await asyncio.sleep(0.02)
    if _first_audio_played_count >= index:
        return "first_audio"
    if _processed_count >= index:
        return "processed"
    return "timeout"


@app.post("/queue")
async def queue_text(request: SynthesizeRequest) -> dict[str, Any]:
    """Add text to the synthesis queue (input streaming / queueing).

    Text is appended to a background queue with its own ``voice`` and
    ``speed``.  A background processor synthesises each queued text
    segment with the voice/speed it was submitted with and the audio
    is played through the server's audio device.  Switching voices
    mid-queue affects only text queued after the switch.

    When ``request.wait`` is true the response blocks until the audio for the
    just-queued item has *started* playing (the moment the playback thread
    writes the first chunk) — not until the whole utterance finishes
    synthesizing, since the audio begins streaming several seconds before
    the last chunk is produced.  The measured latency is included:
    ``synthesis_ms`` (time to generate the audio) and ``latency_ms``
    (Piper-to-speech latency, i.e. time from synthesis start until the
    first audio chunk was spoken; ``n/a`` on a headless server with no
    audio device, where the request falls back to waiting for synthesis to
    complete).

    Args:
        request: Text, voice, and speed parameters.

    Returns:
        Dictionary with queue status and current queue size, plus latency
        info (``synthesis_ms``, ``latency_ms``) when ``request.wait`` is set.

    """
    global _enqueued_count
    resolved_voice = resolve_voice_alias(request.voice)
    logger.info(
        "POST /queue called with text=%r, voice=%s, speed=%.1f",
        request.text[:80],
        resolved_voice,
        request.speed,
    )
    text_queue, audio_queue = await _ensure_queues()
    logger.info(
        "Queues ensured: text_queue=%s, audio_queue=%s",
        text_queue is not None,
        _audio_queue is not None,
    )
    await _start_queue_processor()
    logger.info("Queue processor started/verified")
    loop = asyncio.get_event_loop()
    logger.info("Got event loop: %s", loop)
    _start_playback(loop, audio_queue)
    logger.info("Playback thread started/verified")
    timestamp = time.strftime("%H:%M:%S")
    logger.info(
        "[%s] Message queued: %s (voice=%s, speed=%.1f, queue_size=%d)",
        timestamp,
        request.text[:80],
        resolved_voice,
        request.speed,
        text_queue.qsize(),
    )
    await text_queue.put((request.text, resolved_voice, request.speed))
    _enqueued_count += 1
    my_index = _enqueued_count
    logger.info("Message put into text_queue, new queue_size=%d", text_queue.qsize())

    response: dict[str, Any] = {
        "status": "queued",
        "queue_size": text_queue.qsize(),
    }
    if not request.wait:
        return response

    await _wait_for_first_audio_or_processed(my_index, timeout=_QUEUE_WAIT_TIMEOUT_S)
    # The first-audio path guarantees _last_item_piper_latency_s is set; on
    # the headless fallback path synthesis has completed so the synthesis
    # time is set too.  Either way, give the other thread a brief grace
    # window to publish its values before we report.
    for _ in range(50):
        if _last_item_synthesis_s is not None:
            break
        await asyncio.sleep(0.01)
    response["synthesis_ms"] = (
        round(_last_item_synthesis_s * 1000, 3)
        if _last_item_synthesis_s is not None
        else None
    )
    response["latency_ms"] = (
        round(_last_item_piper_latency_s * 1000, 3)
        if _last_item_piper_latency_s is not None
        else None
    )
    return response


@app.post("/queue/clear")
async def clear_queue() -> dict[str, Any]:
    """Wipe the queue and immediately stop playback.

    Bumps the queue generation (aborting in-flight synthesis and dropping
    any chunks it has not yet delivered), cancels the background queue
    processor so no further items are synthesized, drains all pending text
    and audio, and stops playback so nothing is spoken until new text is
    queued (``POST /queue`` restarts the processor and playback).

    Returns:
        Dictionary with clear status and resulting queue size.

    """
    global _clear_generation
    _clear_generation += 1
    text_queue, audio_queue = await _ensure_queues()
    await _stop_queue_processor()
    while not text_queue.empty():
        text_queue.get_nowait()
    while not audio_queue.empty():
        audio_queue.get_nowait()
    _playback_stop.set()
    return {
        "status": "cleared",
        "queue_size": text_queue.qsize(),
    }


@app.get("/queue/stream")
async def queue_stream() -> StreamingResponse:
    """Stream audio from queued text (output streaming).

    Consumes audio chunks produced by the background queue processor.
    Keep this connection open to receive audio as subsequent
    ``POST /queue`` requests are processed.

    Returns:
        StreamingResponse with raw PCM audio data.

    """
    await _ensure_queues()
    await _start_queue_processor()
    tts = get_or_create_tts()
    sample_rate = tts.sample_rate
    _, audio_queue = await _ensure_queues()
    loop = asyncio.get_event_loop()

    async def _generate() -> AsyncIterator[bytes]:
        while True:
            chunk = await loop.run_in_executor(None, audio_queue.get)
            if chunk is _AUDIO_SENTINEL:
                continue
            yield chunk.audio_int16_bytes

    return StreamingResponse(
        _generate(),
        media_type=f"audio/L16; rate={sample_rate}; channels=1",
    )


# --------------------------------------------------------------------------- #
# Shutdown
# --------------------------------------------------------------------------- #


def _request_server_exit() -> None:
    """Ask the running uvicorn server to shut down gracefully.

    The :func:`serve` entry point stores the active ``uvicorn.Server`` on
    ``app.state.uvicorn_server``.  Setting its ``should_exit`` flag makes
    uvicorn exit its main loop (and stop the process) once the current
    request has been served.  When running under a test client (no uvicorn
    server) this is a no-op.
    """
    server = getattr(app.state, "uvicorn_server", None)
    if server is not None:
        logger.info("Requesting uvicorn server shutdown")
        server.should_exit = True


@app.post("/shutdown")
async def shutdown() -> dict[str, Any]:
    """Tear down the server internally and exit gracefully.

    Stops the queue processor and the playback thread (closing any audio
    sink), drains pending queues, and requests uvicorn to shut down so the
    server process exits.  This is the server half of ``ocr-tts api close``:
    the client sends a ``POST /shutdown`` and the server handles all of its
    own teardown, including any subprocesses it started.

    Returns:
        Dictionary confirming that shutdown has been initiated.

    """
    global _clear_generation
    _clear_generation += 1
    text_queue, audio_queue = await _ensure_queues()
    await _stop_queue_processor()
    while not text_queue.empty():
        text_queue.get_nowait()
    while not audio_queue.empty():
        audio_queue.get_nowait()
    _playback_stop.set()
    _playback_shutdown.set()
    _request_server_exit()
    return {"status": "shutting_down"}


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:  # noqa: S104
    """Run the API server with uvicorn.

    Args:
        host: Bind address.
        port: Listen port.

    """
    import logging

    import uvicorn

    # Configure logging to ensure application INFO logs appear in terminal
    logging.basicConfig(
        level=logging.INFO,
        format="INFO:     %(message)s",
    )

    # Build the server explicitly so the running instance can be stored on
    # ``app.state``; the /shutdown endpoint sets ``server.should_exit`` to
    # trigger a graceful teardown when ``ocr-tts api close`` is invoked.
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    app.state.uvicorn_server = server
    server.run()


if __name__ == "__main__":
    serve()
