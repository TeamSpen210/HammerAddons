from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

from srctools.logger import get_logger

LOGGER = get_logger(__name__)

class Plugin:
    """A plugin loaded by the postcompiler.

    This loads a module, and gives it a logger at __srctools_logger__
    """
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self):
        name = self.path.stem
        spec = spec_from_file_location(name, self.path)

        if not spec:
            raise ValueError('Plugin {} not found at "{}"'.format(name, self.path))

        mod = module_from_spec(spec)

        logname = '.' + name
        #this logger is put into the plugin so it has a consistant log name of "plugin.<plugin name>"
        mod.__srctools_logger__ = get_logger(__name__ + logname, "plugin" + logname)

        spec.loader.exec_module(mod)

        LOGGER.info('Loaded plugin "{}" ({})', name, self.path)
