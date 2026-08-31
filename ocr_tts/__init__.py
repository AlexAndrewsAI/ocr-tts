"""OCR and TTS toolkit.

Provides screen region selection, OCR text extraction,
and text-to-speech capabilities.
"""

import time
from importlib.metadata import version

__version__ = version("ocr-tts")

# Captured at the top of this, the first module Python imports, so it is
# as close to actual process launch as we can get from inside the process.
# Used by the ``--verbose`` latency reporting on ``api send-text`` /
# ``api send-region`` to compute the turnaround time from launch until the
# API returns the synthesis latency.
_launch_monotonic: float = time.monotonic()

__all__ = ["__version__"]
