from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

from srctools.logger import get_logger

LOGGER = get_logger(__name__)

class Plugin:
    def __init__(self, path: Path) -> None:
        self.name = path.stem
        self.path = path

    def load(self):
        spec = spec_from_file_location(self.name, self.path)

        if not spec:
            raise ValueError('Plugin {} not found at "{}"'.format(self.name, self.path))

        mod = module_from_spec(spec)

        logname = '.' + self.name
        mod.__srctools_logger__ = get_logger(__name__ + logname, "plugin" + logname)

        spec.loader.exec_module(mod)

        LOGGER.info('Loaded plugin "{}" ({})', self.name, self.path)
