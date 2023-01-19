"""Postcompiler logic."""
try:
    from ._version import __version__
except ImportError:
    __version__ = '(unknown)'
else:
    # Cleanup and discard module.
    import sys as _sys
    del _sys.modules[_version.__name__]  # type: ignore  # noqa
    del _version, _sys  # type: ignore  # noqa
