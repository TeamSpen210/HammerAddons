import types
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from typing import Optional

from srctools.logger import get_logger

LOGGER = get_logger(__name__)


class Plugin:
    """A plugin loaded by the postcompiler.

    This loads a module, and gives it a logger at LOGGER
    """
    def __init__(self, path: Path) -> None:
        self.path = path
        self.module: Optional[types.ModuleType] = None

    def load(self) -> None:
        """Load and execute the module."""
        name = self.path.stem
        spec = spec_from_file_location(name, self.path)

        if not spec:
            raise FileNotFoundError('Plugin {} not found at "{}"'.format(name, self.path))

        self.module = module_from_spec(spec)

        logname = '.' + name
        # Provide a logger for the plugin, already setup.
        self.module.LOGGER = get_logger(__name__ + logname, "plugin" + logname)

        spec.loader.exec_module(self.module)

        LOGGER.info('Loaded plugin "{}" ({})', name, self.path)
