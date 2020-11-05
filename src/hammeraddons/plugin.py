import types
import sys
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
        name = 'srctools.bsp_tranform_plugin.' + self.path.stem
        spec = spec_from_file_location(name, self.path)

        if not spec:
            raise FileNotFoundError('Plugin file not found at "{}"'.format(self.path.stem, self.path))

        self.module = module_from_spec(spec)

        # Provide a logger for the plugin, already setup.
        self.module.LOGGER = get_logger(name, "plugin:" + self.path.stem)

        spec.loader.exec_module(self.module)
        # Make it importable here - needed for pickling and the like to work.
        sys.modules[name] = self.module

        LOGGER.info('Loaded plugin "{}" from {}', name, self.path)
