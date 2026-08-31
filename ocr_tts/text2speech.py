"""Text-to-Speech using Piper.

Provides a Typer-based CLI and streaming API for converting text to speech
using the Piper TTS engine with automatic voice/model downloading,
streaming audio output, and input queueing.
"""

import logging
import re
import wave
from collections.abc import Iterator
from pathlib import Path

import typer
from piper import AudioChunk, PiperVoice, SynthesisConfig

from ocr_tts import __version__

logger = logging.getLogger(__name__)

__all__ = [
    "PiperTTS",
    "app",
    "download_file",
    "ensure_voice",
    "get_voice_dir",
    "get_voice_urls",
    "parse_voice_name",
]

app = typer.Typer(
    help="Convert text to speech using Piper TTS",
    no_args_is_help=True,
)

DEFAULT_VOICE_DIR = ".piper-voices"
DEFAULT_VOICE = "en_US-hfc_male-medium"

# Pattern for parsing Piper voice names: <lang_family>_<lang_region>-<name>-<quality>
VOICE_PATTERN = re.compile(
    r"^(?P<lang_family>[^-]+)_(?P<lang_region>[^-]+)-"
    r"(?P<voice_name>[^-]+)-(?P<voice_quality>.+)$"
)

# Friendly aliases for commonly used voices
VOICE_ALIASES: dict[str, str] = {
    "male": "en_US-hfc_male-medium",
    "female": "en_US-hfc_female-medium",
}


def resolve_voice_alias(voice: str) -> str:
    """Resolve a friendly voice alias to a Piper voice name.

    Args:
        voice: Voice name or alias (e.g. ``male`` or ``en_US-hfc_male-medium``).

    Returns:
        The resolved Piper voice name.

    """
    return VOICE_ALIASES.get(voice, voice)


# Piper voice download URL format (HuggingFace)
PIPER_DOWNLOAD_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "{lang_family}/{lang_code}/{voice_name}/{voice_quality}/"
    "{lang_code}-{voice_name}-{voice_quality}{extension}?download=true"
)


def download_file(url: str, dest_path: Path) -> None:
    """Download a file from URL to destination path.

    Only HTTP and HTTPS URL schemes are permitted to prevent
    local file access or use of unexpected custom schemes.

    Args:
        url: URL to download from (must use http or https scheme)
        dest_path: Local path to save the file

    Raises:
        typer.Exit: If the URL scheme is not permitted or the
            download fails.

    """
    import urllib.parse
    import urllib.request

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        typer.echo(
            f"Invalid URL scheme '{parsed.scheme}': only http/https allowed",
            err=True,
        )
        raise typer.Exit(code=1) from None

    logger.info("Downloading %s to %s", url, dest_path)
    typer.echo(f"Downloading {dest_path.name}...", err=True)

    try:
        urllib.request.urlretrieve(url, dest_path)  # noqa: S310
        typer.echo(f"Successfully downloaded {dest_path.name}", err=True)
    except Exception as e:
        typer.echo(f"Error downloading {dest_path.name}: {e}", err=True)
        raise typer.Exit(code=1) from None


def get_voice_dir(voice_dir: str | None = None) -> Path:
    """Get the voice directory path, creating it if necessary.

    Args:
        voice_dir: Optional custom voice directory path

    Returns:
        Path to the voice directory

    """
    path = Path(voice_dir) if voice_dir else Path(DEFAULT_VOICE_DIR)

    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_voice_name(voice: str) -> dict[str, str]:
    """Parse a Piper voice name into its components.

    Args:
        voice: Voice name like 'en_US-hfc_male-medium'.

    Returns:
        Dictionary with keys: lang_family, lang_code, voice_name,
            voice_quality.

    Raises:
        ValueError: If the voice name doesn't match the expected pattern.

    """
    match = VOICE_PATTERN.match(voice)
    if not match:
        raise ValueError(
            f"Voice '{voice}' does not match expected pattern "
            "like 'en_US-hfc_male-medium'"
        )
    lang_family = match.group("lang_family")
    lang_region = match.group("lang_region")
    return {
        "lang_family": lang_family,
        "lang_code": f"{lang_family}_{lang_region}",
        "voice_name": match.group("voice_name"),
        "voice_quality": match.group("voice_quality"),
    }


def get_voice_urls(voice: str) -> tuple[str, str]:
    """Get model and config download URLs for a Piper voice.

    Args:
        voice: Voice name like 'en_US-hfc_male-medium'.

    Returns:
        Tuple of (model_url, config_url).

    """
    parts = parse_voice_name(voice)
    model_url = PIPER_DOWNLOAD_URL.format(extension=".onnx", **parts)
    config_url = PIPER_DOWNLOAD_URL.format(extension=".onnx.json", **parts)
    return model_url, config_url


def ensure_voice(voice: str, voice_dir: Path) -> tuple[Path, Path]:
    """Ensure voice model and config files exist, downloading if necessary.

    Args:
        voice: Voice name like 'en_US-hfc_male-medium'.
        voice_dir: Directory where voice files should be stored.

    Returns:
        Tuple of (model_path, config_path).

    """
    model_path = voice_dir / f"{voice}.onnx"
    config_path = voice_dir / f"{voice}.onnx.json"

    if not model_path.exists():
        logger.info("Voice model not found, downloading: %s", model_path)
        model_url, _ = get_voice_urls(voice)
        download_file(model_url, model_path)
    else:
        logger.info("Voice model already exists: %s", model_path)

    if not config_path.exists():
        logger.info("Voice config not found, downloading: %s", config_path)
        _, config_url = get_voice_urls(voice)
        download_file(config_url, config_path)
    else:
        logger.info("Voice config already exists: %s", config_path)

    return model_path, config_path


class PiperTTS:
    """Text-to-speech engine using Piper with streaming support.

    Supports streaming output (audio chunks emitted as sentences are
    processed) and input queueing (accumulating text across calls).

    Attributes:
        voice_name: Name of the Piper voice model.
        voice_dir: Directory containing voice files.

    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        voice_dir: str | None = None,
    ) -> None:
        """Initialize the TTS engine.

        Args:
            voice: Piper voice name or friendly alias (default: en_US-hfc_male-medium).
            voice_dir: Directory for voice files (default: .piper-voices).

        """
        resolved_voice = resolve_voice_alias(voice)
        logger.info(
            "Initializing PiperTTS: voice=%s, voice_dir=%s",
            resolved_voice,
            voice_dir,
        )
        self.voice_name = resolved_voice
        self.voice_dir = get_voice_dir(voice_dir)
        self.model_path, self.config_path = ensure_voice(resolved_voice, self.voice_dir)
        self._voice: PiperVoice | None = None
        logger.info("PiperTTS initialized: voice=%s", voice)

    def _load_voice(self) -> PiperVoice:
        """Load the Piper voice model (cached after first call).

        Returns:
            Loaded PiperVoice instance.

        """
        if self._voice is None:
            logger.info("Loading Piper voice model: %s", self.voice_name)
            self._voice = PiperVoice.load(str(self.model_path), str(self.config_path))
            logger.info("Piper voice model loaded: %s", self.voice_name)
        return self._voice

    @staticmethod
    def _speed_to_length_scale(speed: float) -> float:
        """Convert a speed multiplier to Piper's length_scale.

        Args:
            speed: Speed multiplier (1.0 = normal, 2.0 = 2x faster).

        Returns:
            length_scale value (higher = slower, lower = faster).

        """
        return 1.0 / speed if speed > 0 else 1.0

    @property
    def sample_rate(self) -> int:
        """Get the output sample rate from the loaded voice.

        Returns:
            Sample rate in Hz (default 22050 if voice not yet loaded).

        """
        if self._voice is not None:
            return self._voice.config.sample_rate
        return 22050

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
    ) -> Iterator[AudioChunk]:
        """Synthesize text to speech, yielding audio chunks.

        This is a streaming method — chunks are yielded as each sentence
        is processed, allowing the caller to start using audio before
        all text has been processed.

        Args:
            text: Text to synthesize.
            speed: Speed multiplier (0.5-2.0, default 1.0).

        Yields:
            AudioChunk objects containing raw PCM audio data.

        """
        voice = self._load_voice()
        syn_config = SynthesisConfig(
            length_scale=self._speed_to_length_scale(speed),
        )
        yield from voice.synthesize(text, syn_config)

    def synthesize_to_wav(
        self,
        text: str,
        output: str,
        speed: float = 1.0,
    ) -> None:
        """Synthesize text to a WAV file using streaming writes.

        Audio chunks are written to the WAV file incrementally as they
        are generated, enabling partial output without buffering all
        audio in memory.

        Args:
            text: Text to synthesize.
            output: Output WAV file path.
            speed: Speed multiplier (0.5-2.0, default 1.0).

        """
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        chunks = self.synthesize(text, speed)
        try:
            first_chunk = next(chunks)
        except StopIteration:
            typer.echo("No audio generated from text", err=True)
            return

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(first_chunk.sample_channels)
            wav_file.setsampwidth(first_chunk.sample_width)
            wav_file.setframerate(first_chunk.sample_rate)
            wav_file.writeframes(first_chunk.audio_int16_bytes)

            for chunk in chunks:
                wav_file.writeframes(chunk.audio_int16_bytes)

        typer.echo(f"Successfully created {output}", err=True)


@app.command()
def main(
    text: str = typer.Argument(
        ...,
        help="Text to convert to speech",
    ),
    output: str = typer.Option(
        "output.wav",
        "--output",
        "-o",
        help="Output WAV file path",
    ),
    voice: str = typer.Option(
        DEFAULT_VOICE,
        "--voice",
        "-v",
        help="Voice name or alias (e.g., en_US-hfc_male-medium, male, female)",
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
        help="Speech speed multiplier (0.5-2.0)",
    ),
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit",
        is_eager=True,
    ),
) -> None:
    """Convert text to speech and save as WAV file.

    Uses the Piper TTS engine for high-quality, offline speech synthesis.
    Audio chunks are streamed to the output file as they are generated.

    Example:
        text2speech "Hello, world!" -o greeting.wav
        text2speech "Hello" -v en_US-hfc_male-medium -o out.wav
        text2speech "Bonjour" -v fr_FR-siwis-medium -o french.wav

    """
    if version:
        typer.echo(f"text2speech {__version__}")
        raise typer.Exit()

    typer.echo(
        f"Generating speech for: {text[:50]}{'...' if len(text) > 50 else ''}",
        err=True,
    )
    try:
        tts = PiperTTS(voice=voice, voice_dir=voice_dir)
        typer.echo(f"Loading Piper voice '{voice}'...", err=True)
        tts.synthesize_to_wav(text, output, speed)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error generating speech: {e}", err=True)
        logger.exception("Failed to generate speech")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
