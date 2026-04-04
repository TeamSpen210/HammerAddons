"""Postcompiler logic."""
from pathlib import Path
import sys


__all__ = ['BINS_PATH', 'FROZEN', 'HADDONS_VER', 'SRCTOOLS_VER', 'WIN', 'MAC', 'LINUX']


try:
    from ._version import HADDONS_VER, SRCTOOLS_VER
except ImportError:
    HADDONS_VER = SRCTOOLS_VER = '(unknown)'
else:
    # Cleanup and discard module.
    import sys as _sys
    del _sys.modules[_version.__name__]  # type: ignore  # noqa
    del _version, _sys  # type: ignore  # noqa


try:
    # PyInstaller sets this attribute.
    BINS_PATH = Path(sys._MEIPASS)  # type: ignore  # noqa
    FROZEN = True
except AttributeError:
    # Root directory is up thrice from postcompiler.py.
    BINS_PATH = Path(sys.argv[0], '..', '..', '..').resolve()
    FROZEN = False

WIN = sys.platform.startswith('win')
MAC = sys.platform.startswith('darwin')
LINUX = sys.platform.startswith('linux')
