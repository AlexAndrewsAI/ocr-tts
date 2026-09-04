"""Real-time streaming playback of synthesized speech.

This module wires the streaming :class:`ocr_tts.text2speech.PiperTTS`
engine to an audio output device so that text is spoken aloud
*as it is synthesized*, and additional text can be enqueued at any time
from any thread.

The pattern is a classic producer/buffer/consumer split:

* **Producer** — :meth:`StreamingPlayer.say` appends text to a FIFO
  queue.  It can be called at arbitrary times, from any thread, and each
  call simply appends to the queue.
* **Synthesis worker** — a background thread drains the text queue and
  converts each item into audio chunks via ``PiperVoice.synthesize()``,
  which already yields one chunk per sentence.  Chunks are forwarded to
  an audio buffer as soon as they are produced.
* **Playback worker** — a background thread pulls chunks from the audio
  buffer and writes them to an :class:`AudioSink` the instant they
  arrive, so the first sentence starts playing before later text has
  even been synthesized.

Typical usage::

    streamer = StreamingPlayer(tts=PiperTTS(), sink=SounddeviceSink())
    streamer.start()
    streamer.say("First sentence starts playing immediately.")
    # ... later, from another thread or request handler ...
    streamer.say("More text, appended to the queue.")
    streamer.stop()

The audio sink is pluggable so the module can be tested in headless CI
with a fake sink and does not hard-depend on a sound card.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from types import TracebackType
from typing import Any, Protocol

import typer
from piper import AudioChunk

from ocr_tts.text2speech import DEFAULT_VOICE, PiperTTS

logger = logging.getLogger(__name__)

__all__ = [
    "AudioSink",
    "SounddeviceSink",
    "StreamingPlayer",
    "Synthesizer",
    "app",
]

# Maximum time (seconds) to block waiting for queue items / thread exit.
_POLL_TIMEOUT_S = 0.1

# Time (seconds) to wait for a worker thread to exit during start()/stop().
_JOIN_TIMEOUT = 30.0


class Synthesizer(Protocol):
    """Structural protocol for anything that can synthesize text to audio.

    :class:`ocr_tts.text2speech.PiperTTS` satisfies this protocol, as do
    lightweight test doubles.
    """

    def synthesize(self, text: str, speed: float = 1.0) -> Iterator[AudioChunk]:
        """Yield one audio chunk per sentence of ``text``.

        Args:
            text: Text to synthesize.
            speed: Speed multiplier (1.0 = normal).

        Yields:
            Streaming audio chunks.

        """


class AudioSink(ABC):
    """Pluggable destination for 16-bit PCM audio samples.

    An :class:`AudioSink` receives raw PCM bytes and forwards them to an
    audio device.  Implementations are expected to be called from a
    single playback thread, in FIFO order.

    The interface is deliberately small so that real hardware drivers
    (e.g. PortAudio via :mod:`sounddevice`) and test doubles share the
    same contract.
    """

    @abstractmethod
    def open(self, sample_rate: int, channels: int, sample_width: int) -> None:
        """Open the output device with the given audio format.

        Args:
            sample_rate: Samples per second (Hz).
            channels: Number of audio channels.
            sample_width: Bytes per sample (2 for 16-bit PCM).

        """

    @abstractmethod
    def write(self, pcm: bytes) -> None:
        """Write a block of raw PCM to the device.

        Args:
            pcm: Raw PCM bytes (interpreted per the format given to
                :meth:`open`).

        """

    @abstractmethod
    def close(self) -> None:
        """Flush and close the output device."""


class SounddeviceSink(AudioSink):
    """Play 16-bit PCM through the system audio device via sounddevice.

    The ``sounddevice`` dependency is imported lazily inside
    :meth:`open`, so merely importing this module (or using a different
    :class:`AudioSink`) never requires a working PortAudio installation.
    """

    def __init__(self, blocksize: int = 1024) -> None:
        """Initialize the sounddevice sink.

        Args:
            blocksize: Frames per buffer written to the device.

        """
        self._blocksize = blocksize
        self._stream: Any = None

    def open(self, sample_rate: int, channels: int, _sample_width: int) -> None:
        """Open a low-latency 16-bit PCM output stream.

        Args:
            sample_rate: Samples per second (Hz).
            channels: Number of audio channels.
            _sample_width: Bytes per sample (2 for 16-bit PCM).

        """
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            blocksize=self._blocksize,
            dtype="int16",
            latency="low",
        )
        self._stream.start()

    def write(self, pcm: bytes) -> None:
        """Write a raw 16-bit PCM block to the device.

        Args:
            pcm: Raw 16-bit PCM bytes.

        Raises:
            RuntimeError: If the sink has not been opened.

        """
        import numpy as np

        if self._stream is None:
            raise RuntimeError("Sink not open; call open() before write()")
        # Convert bytes to numpy int16 array for sounddevice
        audio_data = np.frombuffer(pcm, dtype=np.int16)
        self._stream.write(audio_data)

    def close(self) -> None:
        """Flush remaining audio and close the output stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class StreamingPlayer:
    """Stream text to a TTS engine and play audio back immediately.

    Text submitted via :meth:`say` is appended to an internal FIFO queue.
    A synthesis thread converts queued text into audio chunks (streaming
    per sentence) and a playback thread forwards each chunk to an
    :class:`AudioSink` the moment it is ready, so speech starts before
    the entire text has been synthesized.  Submitting more text at any
    time, from any thread, simply appends to the queue.

    Attributes:
        tts: The TTS engine used to synthesize queued text.
        sink: The audio device used to play synthesized chunks.

    """

    def __init__(self, tts: Synthesizer, sink: AudioSink) -> None:
        """Initialize the player.

        Args:
            tts: Object with a streaming ``synthesize`` method.
            sink: Audio sink that consumes the synthesized PCM.

        """
        self.tts = tts
        self.sink = sink
        # Queue items are (text, speed) tuples; ``None`` marks shutdown.
        self._text_queue: queue.Queue[tuple[str, float] | None] = queue.Queue()
        # Queue items are AudioChunk; ``None`` marks end of all audio.
        self._audio_queue: queue.Queue[AudioChunk | None] = queue.Queue()
        self._started = False
        self._error: Exception | None = None
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> StreamingPlayer:
        """Start the player when used as a context manager."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Gracefully stop the player on context exit."""
        self.stop()

    def start(self) -> None:
        """Start the background synthesis and playback threads.

        Calling this more than once is a no-op.  If a previous
        :meth:`stop` could not join a thread (it was still draining the
        shared queues), start defends against spawning fresh threads on
        top of the old ones by joining any straggler first; the player
        stays stopped if one cannot exit.  Any error from a previous run
        is cleared on restart.
        """
        if self._started:
            return
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=_JOIN_TIMEOUT)
                if thread.is_alive():
                    logger.warning(
                        "Thread '%s' still alive from a previous run; "
                        "refusing to start the player",
                        thread.name,
                    )
                    return
        self._error = None
        self._started = True
        synthesis = threading.Thread(
            target=self._synthesis_loop,
            name="tts-synthesis",
            daemon=True,
        )
        playback = threading.Thread(
            target=self._playback_loop,
            name="tts-playback",
            daemon=True,
        )
        self._threads = [synthesis, playback]
        synthesis.start()
        playback.start()
        logger.info("Streaming player started")

    def say(self, text: str, speed: float = 1.0) -> int:
        """Enqueue text to be synthesized and spoken.

        Safe to call from any thread, at any time; each call simply
        appends to the FIFO queue after any pending text.

        Args:
            text: Text to speak.
            speed: Speed multiplier (1.0 = normal).

        Returns:
            Number of text items currently waiting in the queue.

        """
        self._text_queue.put((text, speed))
        return self._text_queue.qsize()

    def pending(self) -> int:
        """Return the number of text items not yet synthesized."""
        return self._text_queue.qsize()

    def stop(self) -> None:
        """Stop the player, finishing all queued text before returning.

        Graceful shutdown: queued text is fully synthesized and played
        before the threads exit and the sink is closed.  Only when every
        thread has actually exited is the player marked stopped; if a
        thread is still draining after the join timeout, the player stays
        marked as running so a later :meth:`start` cannot spawn duplicate
        threads alongside the stragglers.
        """
        if not self._started:
            return
        self._text_queue.put(None)
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=_JOIN_TIMEOUT)
        if any(thread.is_alive() for thread in self._threads):
            logger.warning(
                "Some player threads did not stop cleanly; "
                "player remains marked as running"
            )
            return
        self.sink.close()
        self._started = False
        logger.info("Streaming player stopped")

    def _synthesis_loop(self) -> None:
        """Background worker: drain text queue, synthesize, forward audio."""
        try:
            while True:
                item = self._text_queue.get()
                if item is None:
                    self._audio_queue.put(None)
                    return
                text, speed = item
                for chunk in self.tts.synthesize(text, speed):
                    self._audio_queue.put(chunk)
        except Exception as exc:
            logger.exception("Synthesis loop failed")
            self._error = exc
            self._audio_queue.put(None)

    def _playback_loop(self) -> None:
        """Background worker: drain audio queue, write chunks to the sink."""
        opened = False
        try:
            while True:
                item = self._audio_queue.get()
                if item is None:
                    return
                chunk = item
                if not opened:
                    self.sink.open(
                        chunk.sample_rate,
                        chunk.sample_channels,
                        chunk.sample_width,
                    )
                    opened = True
                self.sink.write(chunk.audio_int16_bytes)
        except Exception as exc:
            logger.exception("Playback loop failed")
            self._error = exc


app = typer.Typer(
    help="Stream text to TTS and play it back immediately (live mode)",
    no_args_is_help=True,
)


@app.command()
def live(
    voice: str = typer.Option(
        DEFAULT_VOICE,
        "--voice",
        "-v",
        help="Voice name (e.g., en_US-hfc_male-medium)",
    ),
    voice_dir: str | None = typer.Option(
        None,
        "--voice-dir",
        help="Directory for voice/model files (default: .piper-voices)",
    ),
    speed: float = typer.Option(
        1.0,
        "--speed",
        "-s",
        min=0.1,
        max=3.0,
        help="Speech speed multiplier (0.1-3.0, default: 1.0)",
    ),
) -> None:
    """Read lines from stdin and speak each one immediately.

    Each line of stdin is enqueued as soon as it is read; a line is
    spoken as soon as the TTS engine has produced its first audio chunk,
    without waiting for input to finish.
    """
    try:
        tts = PiperTTS(voice=voice, voice_dir=voice_dir)
        typer.echo(f"Loading Piper voice '{voice}'...", err=True)
        with StreamingPlayer(tts, SounddeviceSink()) as player:
            for raw in sys.stdin:
                line = raw.rstrip("\n")
                if line.strip():
                    player.say(line, speed)
    except Exception as exc:
        typer.echo(f"Error in live playback: {exc}", err=True)
        logger.exception("Live playback failed")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
