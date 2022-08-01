"""Handles user configuration common to the different scripts."""
from pathlib import Path
from typing import Tuple, Set, Dict
import sys

from atomicwrites import atomic_write

from srctools.game import Game
from srctools import Property, logger
from srctools.filesys import FileSystemChain, FileSystem, RawFileSystem, VPKFileSystem

from .props_config import Opt, Config, TYPE
from .plugin import Source as PluginSource, PluginFinder, BUILTIN as BUILTIN_PLUGIN


__all__ = [
    'LOGGER',
    'parse',
    'OPTIONS',
]

LOGGER = logger.get_logger(__name__)
CONF_NAME = 'srctools.vdf'


def parse(path: Path, game_folder: str='') -> Tuple[
    Config,
    Game,
    FileSystemChain,
    Set[FileSystem],
    PluginFinder,
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
        * The plugin loader.
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
            if folder.parent.stem in ('common', 'sourcemods'):
                break
        else:
            # Give up, put next to the input path.
            folder = path.parent
        conf.path = folder / CONF_NAME

        LOGGER.warning('Writing default to "{}"', conf.path)

    with atomic_write(conf.path, overwrite=True) as f:
        conf.save(f)

    if not game_folder:
        game_folder = conf.get(str, 'gameinfo')
    if not game_folder:
        raise ValueError(
            'No game folder specified!\n'
            'Add -game $gamedir to the command line, or set it in '
            f'"{conf.path}".'
        )
    game = Game((folder / game_folder).resolve())
    LOGGER.info('Game folder: {}', game.path)

    fsys_chain = game.get_filesystem()

    blacklist: Set[FileSystem] = set()

    if not conf.get(bool, 'pack_vpk'):
        for fsys, prefix in fsys_chain.systems:
            if isinstance(fsys, VPKFileSystem):
                blacklist.add(fsys)

    game_root = game.root

    for prop in conf.get(Property, 'searchpaths'):
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

    sources: Dict[str, PluginSource] = {}

    if hasattr(sys, 'frozen'):
        builtin_transforms = (Path(sys.executable).parent / 'transforms').resolve()
    else:
        # Assume working directory is HammerAddons.
        builtin_transforms = Path('transforms').resolve()

    # Find all the plugins and make plugin objects out of them
    unnamed_ind = 1
    for prop in conf.get(Property, 'plugins'):
        source = PluginSource.parse(game_root, prop)
        if not source.id:
            source.id = f'unnamed_{unnamed_ind}'
            unnamed_ind += 1
        if source.id in sources:
            raise ValueError(f'Plugin "{source.id}" declared twice!')
        sources[source.id] = source

    if BUILTIN_PLUGIN not in sources:
        sources[BUILTIN_PLUGIN] = PluginSource(BUILTIN_PLUGIN, builtin_transforms, recursive=True)

    for source in sources.values():
        LOGGER.debug('- {!r}', source)

    plugin_finder = PluginFinder('hammeraddons.plugins', sources)
    sys.meta_path.append(plugin_finder)

    return conf, game, fsys_chain, blacklist, plugin_finder


OPTIONS = [
    Opt(
        'gameinfo', TYPE.STR,
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
        """Allow files in VPKs to be packed into the map. 
        This is disabled by default since these are usually default files.
    """),
    Opt(
        'pack_dump', TYPE.STR,
        """If set, copy all the packed resoures to this additional location.
        You can also prefix this with a # character to only copy to this 
        destination, not the BSP pakfile.
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
        'soundscript_manifest', False,
        """Generate and pack game_sounds_manifest.txt, with all used soundscripts.     
        This is needed to make packing soundscripts work for the Portal 2 
        workshop.
        """,
    ),
    Opt(
        'particles_manifest', '',
        """If set to a path, generate and pack a particles manifest under this name.     
        This is needed to make packing particles work. "<map name>" is replaced with the map name.
        Depending on your game, these are some of the correct paths:
        * particles/particles_manifest.txt
        * maps/<map name>_particles.txt (TF2)
        * particles/<map name>_manifest.txt (L4D2, Portal 2)
        """,
    ),
    Opt(
        'studiomdl', 'bin/studiomdl.exe',
        """Set the path to StudioMDL so the compiler can generate props.
        If blank these features are disabled.
        This is relative to the game root.
        """,
    ),
    Opt(
        'use_comma_sep', TYPE.BOOL,
        """Before L4D, entity I/O used ',' to seperate the different parts.
    
       Later games used a special symbol to delimit the sections, allowing
       commas to be used in outputs. The compiler will guess which to use
       based on existing outputs in the map, but if this is incorrect 
       (or if there aren't any in the map), use this to override.
    """),

    Opt(
        'propcombine_qc_folder', Property('', [Property('loc', '../content')]),
        """Define where the QC files are for combinable static props.
        This path is searched recursively. This defaults to 
        the 'content/' folder, which is adjacent to the game root.
        This is how Valve sets up their file structure.
        """
    ),
    Opt(
        'propcombine_crowbar', True,
        """If enabled, Crowbar will be used to decompile models which don't have
        a QC in the provided QC folder.
        """
    ),
    Opt(
        'propcombine_cache', "decomp_cache/",
        """Cache location for models decompiled for combining."""
    ),
    Opt(
        'propcombine_volume_tolerance', -1.0,
        """When propcombining, an attempt will be made to merge collision meshes.
        
        If shrink wrapping a pair of meshes changes the volume less than this,
        the combined version will be used. If negative, this will not be done.
        """
    ),
    Opt(
        'propcombine_auto_range', 0,
        """If greater than zero, combine props at least this close together.""",
    ),
    Opt(
        'propcombine_min_cluster', 2,
        """The minimum number of props required before propcombine will
        bother merging them. Should be greater than 1.
        """,
    ),
    Opt(
        'propcombine_blacklist', Property('', []),
        """Models specified here will never be propcombined.
        
        You can specify a full path, or one with * wildcards. Alternatively,
        set 'no_propcombine' in the model $keyvalues.
        """,
    ),
    Opt(
        'propcombine_pack', True,
        """If set, force-pack the combined props."""
    ),
    Opt(
        'plugins', TYPE.RAW,
        """\
        Add plugins to the post compiler. Each block is a package of plugins in some folder.
        The name must be a Python identifier - the plugins are mounted at "hammeraddons.bsp_transforms.plugin.blockname.filename".
        * "path" must be set to either a single Python file, or a folder of files.
        * If "recurse" is set, subfolders are recursively loaded as packages.
        The transforms folder inside the postcompiler folder is also always
        loaded, under the name "builtin".
    """),
    Opt(
        'transform_opts', TYPE.RAW,
        """Specify additional options specific to transforms. Each key here is the name of the 
        transform, and the value is then decided by that transform.
        """
    ),
]
