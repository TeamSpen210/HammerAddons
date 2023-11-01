"""Postcompiler logic."""
from pathlib import Path
import sys


try:
    from ._version import __version__
except ImportError:
    __version__ = '(unknown)'
else:
    # Cleanup and discard module.
    import sys as _sys
    del _sys.modules[_version.__name__]  # type: ignore  # noqa
    del _version, _sys  # type: ignore  # noqa


try:
    # PyInstaller sets this attribute.
    BINS_PATH = Path(sys._MEIPASS)  # noqa
    FROZEN = True
except AttributeError:
    # Root directory is up thrice from postcompiler.py.
    BINS_PATH = Path(sys.argv[0], '..', '..', '..').resolve()
    FROZEN = False
