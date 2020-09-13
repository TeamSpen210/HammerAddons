"""Handles user configuration common to the different scripts."""
from pathlib import Path
from typing import Tuple, Set

from srctools.game import Game

from srctools import Property, logger, AtomicWriter
from srctools.filesys import FileSystemChain, FileSystem, RawFileSystem, VPKFileSystem
from srctools.props_config import Opt, Config, TYPE

from srctools.scripts.plugin import Plugin

__all__ = [
    'LOGGER',
    'parse',
    'OPTIONS',
]

LOGGER = logger.get_logger(__name__)

CONF_NAME = 'srctools.vdf'


def parse(path: Path) -> Tuple[
    Config,
    Game,
    FileSystemChain,
    Set[FileSystem],
    Set[Plugin],
]:
    """From some directory, locate and parse the config file.

    This then constructs and customises each object according to config
    options.

    The first srctools.vdf file found in a parent directory is parsed.
    If none can be found, it tries to find the first subfolder of 'common/' and
    writes a default copy there. FileNotFoundError is raised if none can be
    found.

    This returns:
        * The config.
        * Parsed gameinfo.
        * The chain of filesystems.
        * A packing blacklist.
        * A list of plugins.
    """
    conf = Config(OPTIONS)

    # If the path is a folder, add a dummy folder so parents yields it.
    # That way we check for a config in this folder.
    if not path.suffix:
        path /= 'unused'

    for folder in path.parents:
        conf_path = folder / CONF_NAME
        if conf_path.exists():
            LOGGER.info('Config path: "{}"', conf_path.absolute())
            with open(conf_path) as f:
                props = Property.parse(f, conf_path)
            conf.path = conf_path
            conf.load(props)
            break
    else:
        LOGGER.warning('Cannot find a valid config file!')
        # Apply all the defaults.
        conf.load(Property(None, []))

        # Try to write out a default file in the game folder.
        for folder in path.parents:
            if folder.parent.stem == 'common':
                break
        else:
            # Give up, write to working directory.
            folder = Path()
        conf.path = folder / CONF_NAME

        LOGGER.warning('Writing default to "{}"', conf.path)

        with AtomicWriter(str(conf.path)) as f:
            conf.save(f)

    game = Game((folder / conf.get(str, 'gameinfo')).resolve())

    fsys_chain = game.get_filesystem()

    blacklist = set()  # type: Set[FileSystem]

    if not conf.get(bool, 'pack_vpk'):
        for fsys, prefix in fsys_chain.systems:
            if isinstance(fsys, VPKFileSystem):
                blacklist.add(fsys)

    game_root = game.root

    for prop in conf.get(Property, 'searchpaths'):  # type: Property
        if prop.has_children():
            raise ValueError('Config "searchpaths" value cannot have children.')
        assert isinstance(prop.value, str)

        if prop.value.endswith('.vpk'):
            fsys = VPKFileSystem(str((game_root / prop.value).resolve()))
        else:
            fsys = RawFileSystem(str((game_root / prop.value).resolve()))

        if prop.name in ('prefix', 'priority'):
            fsys_chain.add_sys(fsys, priority=True)
        elif prop.name == 'nopack':
            blacklist.add(fsys)
        elif prop.name in ('path', 'pack'):
            fsys_chain.add_sys(fsys)
        else:
            raise ValueError(
                'Unknown searchpath '
                'key "{}"!'.format(prop.real_name)
            )

    plugins = set()  # type: Set[Plugin]

    for prop in conf.get(Property, 'plugins'):  # type: Property
        if prop.has_children():
            raise ValueError('Config "plugins" value cannot have children.')
        assert isinstance(prop.value, str)
        
        plugins.add(Plugin(prop.name, game_root / Path(prop.value)))

    return conf, game, fsys_chain, blacklist, plugins


OPTIONS = [
    Opt(
        'gameinfo', 'portal2/',
        """The main game folder. portal2/ for Portal 2, csgo/ for CSGO, etc.
        This is relative to the config file.
    """),
    Opt(
        'auto_pack', True,
        """Automatically find and pack files in the map. 
        If this is disabled, specifically-indicated files will still be 
        added as well as their dependencies.
    """),
    Opt(
        'pack_vpk', False,
        """Prevent files in VPKs from being packed into the map.
    """),
    Opt(
        'searchpaths', TYPE.RAW,
        """\
        Add additional search paths to the game. Each key-value pair
        defines a path, with the value either a folder path or a VPK 
        filename relative to the game root. The key defines the behaviour:
        * "prefix" "folder/" adds the path to the start, so it overrides
            all others.
        * "path" "vpk_path.vpk" adds the path to the end, so it is checked last.
        * "nopack" "folder/" prohibits files in this path from being packed, you'll need to use one of the others also to add the path.
    """),
    Opt(
        'studiomdl', '',
        """Set the path to StudioMDL so the compiler can generate props.
        If unset these features are disabled.
        This is relative to the game root.
        """,
        fallback='propcombine_studiomdl',
    ),

    Opt(
        'propcombine_studiomdl', '',
        """Old name for "studiomdl".
        """,
    ),
    Opt(
        'propcombine_qc_folder', Property('', []),
        """Define where the QC files are for combinable static props.
        This path is searched recursively. If unset this defaults to 
        the 'content/' folder, which is adjacent to the game root.
        This is how Valve sets up their file structure.
        """
    ),
    Opt(
        'propcombine_auto_range', 0,
        """If greater than zero, combine props at least this close together.
        This is ignored if comp_propcombine_set entities are in the map.
        """,
    ),
    Opt(
        'propcombine_min_cluster', 2,
        """The minimum number of props required before propcombine will
        bother merging them. Should be greater than 1.
        """,
    ),
    Opt(
        'plugins', TYPE.RAW,
        """Plugins to load. The key is the module name while the value is the path to it.
        """,
    ),
]
