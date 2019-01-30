"""Handles user configuration common to the different scripts."""
from pathlib import Path

from srctools import Property, logger, AtomicWriter
from srctools.props_config import Opt, Config, TYPE


__all__ = [
    'LOGGER',
    'find_conf',
    'OPTIONS',
]

LOGGER = logger.get_logger(__name__)

CONF_NAME = 'srctools.vdf'


def find_conf(path: Path) -> 'Config':
    """From some directory, locate and parse the config file.

    The first srctools.vdf file found in a parent directory is parsed.
    If none can be found, it tries to find the first subfolder of 'common/' and
    writes a default copy there. FileNotFoundError is raised if none can be
    found.
    """
    conf = Config(OPTIONS)

    for folder in path.parents:
        conf_path = folder / CONF_NAME
        if conf_path.exists():
            LOGGER.info('Config path: "{}"', conf_path.absolute())
            with open(conf_path) as f:
                props = Property.parse(f, conf_path)
            conf.load(props)
            break
    else:
        LOGGER.warning('Cannot find a valid config file!')
        # Try to write out a default file in the game folder.
        for folder in path.parents:
            if folder.parent.stem == 'common':
                break
        else:
            # Give up, write to working directory.
            folder = Path()
        path = str(folder / CONF_NAME)

        LOGGER.warning('Writing default to "{}"', path)

        with AtomicWriter(path) as f:
            conf.save(f)

    return conf


OPTIONS = [
    Opt(
        'gameinfo', 'portal2/',
        """The main game folder. portal2/ for Portal 2, csgo/ for CSGO, etc.
    """),
    Opt(
        'pack_vpk', False,
        """Prevent files in VPKs from being packed into the map.
    """),
    Opt(
        'searchpaths', TYPE.PROP,
        """\
        Add additional search paths to the game. Each key-value pair
        defines a path, with the value either a folder path or a VPK 
        filename. The key defines the behaviour:
        * "prefix" "folder/" adds the path to the start, so it overrides
            all others.
        * "path" "vpk_path.vpk" adds the path to the end, so it is checked last.
        * "nopack" "folder/" prohibits files in this path from being packed.
    """),
]
