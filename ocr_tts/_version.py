"""Package version resolution.

``importlib.metadata.version`` raises ``PackageNotFoundError`` when the
distribution is not installed (e.g. bare source-tree use or a partially
failed editable install in CI), so the lookup is guarded and falls back to
a hard-coded constant.  Keeping this in its own module means importing any
:mod:`ocr_tts` submodule never hard-fails at import time.
"""

from importlib.metadata import PackageNotFoundError, version

_DEFAULT_VERSION = "0.1.0"

try:
    __version__ = version("ocr-tts")
except PackageNotFoundError:
    __version__ = _DEFAULT_VERSION
