"""Handles user configuration common to the different scripts."""
from typing import Callable, Dict, Iterator, Optional, Pattern as re_Pattern, Set, Union, Final
from typing_extensions import TypeAlias
from pathlib import Path
import fnmatch
import re
import sys

from srctools import AtomicWriter, Keyvalues, conv_int, logger
from srctools.filesys import FileSystem, FileSystemChain, RawFileSystem, VPKFileSystem
from srctools.game import Game
import attrs

from .plugin import BUILTIN as BUILTIN_PLUGIN, PluginFinder, Source as PluginSource
from .props_config import Opt, Options

from srctools.steam import find_app

LOGGER = logger.get_logger(__name__)
CONF_NAME: Final = 'srctools.vdf'
PATHS_NAME: Final = 'srctools_paths.vdf'

PATH_KEY_GAME: Final = 'gameinfo_path'
PATH_KEY_MAP: Final = 'mapdir_path'

PREDEFINED_PATHS = {PATH_KEY_GAME, PATH_KEY_MAP}

# Matches cubemap files. Put here, so we can write it into the docstring.
CUBEMAP_REGEX = r"materials/maps/.*/(c[0-9-]+_[0-9-]+_[0-9-]+|cubemapdefault)(\.hdr)?\.vtf"

# Tags we use in our engine dump.
USED_PACK_TAGS: Set[str] = {
    'hl1', 'hl2', 'episodic',
    'tf2',
    'mapbase', 'entropyzero2',
    'mesa', 'p2',
}

PATHS_CONF_STARTER: Final = f'''\
// This config contains a list of directories which can be referenced by the main config.
// Keeping this a separate file allows the main config to be shared in a mod team, while this
// config is customised for each user's installation locations.
// The keys here are then referenced by specifying "|key|" at the start of a path.
// If no root is specified, paths are relative to these configs.
// Some names are predefined: |{PATH_KEY_GAME}| and |{PATH_KEY_MAP}|.
"Paths"
    {{
    // For example this makes "|hl2|/episodic/ep1_pak_dir.vpk" valid in searchpaths.
    // "hl2" "C:/Program Files/Steam/SteamApps/common/Half Life 2/"
    }}
'''
# A function taking a configured path, and expanding |refs| to get the full location.
Expander: TypeAlias = Callable[[str], Path]


def make_expander(roots: Dict[str, Path], orig_root: Union[str, Path]) -> Expander:
    """Produce a function that expands configs potentially containing || refs."""
    def expander(path: str) -> Path:
        """Expand a reference potentially containing || refs."""
        root = orig_root
        if path.startswith('|'):
            _, ref, path = path.split('|', 2)
            path = path.lstrip('\\/')  # Make |loc|/blah/ allowed, don't treat as a root.
            try:
                root = roots[ref.casefold()]
            except KeyError:
                LOGGER.warning(
                    '|{}| is not defined in {}! Assuming {}\nKnown: {}',
                    ref, PATHS_NAME, root,
                    ', '.join(sorted(roots)),
                )
        return Path(root, path).resolve()
    return expander


@attrs.frozen(kw_only=True)
class Config:
    """Result of parse()."""
    opts: Options
    game: Game
    fsys: FileSystemChain
    pack_blacklist: Set[FileSystem]
    plugins: PluginFinder
    expand_path: Expander

    @property
    def loc(self) -> Path:
        """Location of the configs."""
        path = self.opts.path
        assert path is not None
        return path


def parse(map_path: Path, game_folder: Optional[str]='') -> Config:
    """From some directory, locate and parse the config files.

    This then constructs and customises each object according to config
    options.

    The first srctools.vdf file found in a parent directory is parsed.
    If none can be found, it tries to find the first subfolder of 'common/' and
    writes a default copy there. FileNotFoundError is raised if none can be
    found.
    """
    opts = Options(globals())

    # If the path is a folder, add a dummy folder so parents yields it.
    # That way we check for a config in this folder.
    if not map_path.suffix:
        map_path /= 'unused'

    for folder in map_path.parents:
        conf_path = folder / CONF_NAME
        if conf_path.exists():
            LOGGER.info('Config path: "{}"', conf_path.absolute())
            with open(conf_path, encoding='utf8') as f:
                kv = Keyvalues.parse(f, conf_path)
            opts.path = conf_path
            opts.load(kv)
            break
    else:
        LOGGER.warning('Cannot find a valid config file!')
        # Apply all the defaults.
        opts.load(Keyvalues(None, []))

        # Try to write out a default file in the game folder.
        for folder in map_path.parents:
            if folder.parent.stem in ('common', 'sourcemods'):
                break
        else:
            # Give up, put next to the input path.
            folder = map_path.parent
        opts.path = folder / CONF_NAME

        LOGGER.warning('Writing default to "{}"', opts.path)

    # Add in new pack tags to the config.
    pack_tags = opts.get(PACK_TAGS)
    for tag in USED_PACK_TAGS:
        if tag not in pack_tags:
            pack_tags[tag] = '0'
    opts.set_opt(PACK_TAGS, pack_tags)

    with AtomicWriter(opts.path) as f:
        opts.save(f)

    # Fetch the additional path config.
    path_roots: Dict[str, Path] = {}
    paths_conf_loc = opts.path.with_name(PATHS_NAME)
    LOGGER.info('Paths config: {}', paths_conf_loc)
    try:
        with open(paths_conf_loc, encoding='utf8') as f:
            for kv in Keyvalues.parse(f).find_children('Paths'):
                if kv.has_children():
                    LOGGER.warning('Paths configs may not be blocks!')
                else:
                    name = kv.name.strip('|')
                    if name in PREDEFINED_PATHS:
                        LOGGER.warning(
                            '|{}| cannot be defined in the path config - '
                            'the following names are builtin: {}',
                            kv.name, sorted(PREDEFINED_PATHS),
                        )
                    path_roots[name] = Path(kv.value)
    except FileNotFoundError:
        with open(paths_conf_loc, 'w', encoding='utf8') as f:
            f.write(PATHS_CONF_STARTER)

    if not game_folder:
        game_folder = opts.get(GAMEINFO)
    if not game_folder:
        raise ValueError(
            'No game folder specified!\n'
            'Add -game $gamedir to the command line, or set it in '
            f'"{opts.path}".'
        )

    expand_path = make_expander(path_roots, folder)
    game = Game(expand_path(game_folder))
    LOGGER.info('Game folder: {}', game.path)
    # Now we located it, other definitions can use this loc.
    path_roots[PATH_KEY_GAME] = game.path
    path_roots[PATH_KEY_MAP] = map_path.parent

    fsys_chain = game.get_filesystem()

    blacklist: Set[FileSystem] = set()

    if not opts.get(PACK_VPK):
        for fsys, prefix in fsys_chain.systems:
            if isinstance(fsys, VPKFileSystem):
                blacklist.add(fsys)

    for kv in opts.get(SEARCHPATHS):
        if kv.has_children():
            raise ValueError('Config "searchpaths" value cannot have children.')
        assert isinstance(kv.value, str)

        appid = -1
        # Game mount, we just replace the <appid> with a path, this will ensure compatibility with .vpk
        if (end := kv.value.find(">")) and kv.value.startswith("<"):
            appid = conv_int(kv.value[1:end])

        if appid != -1:
            LOGGER.info("Mounting appid {}", appid)
            try:
                info = find_app(appid)
            except KeyError:
                LOGGER.warning("No game with appid {} found!", appid)
            else:
                LOGGER.info(f"Mounted game {info.name} with path: {info.path}")
                kv.value = (info.path / kv.value[end + 1:]).as_posix()

        if kv.value.endswith('.vpk'):
            fsys = VPKFileSystem(str(expand_path(kv.value)))
        else:
            fsys = RawFileSystem(str(expand_path(kv.value)))

        if kv.name in ('prefix', 'priority'):
            fsys_chain.add_sys(fsys, priority=True)
        elif kv.name == 'nopack':
            blacklist.add(fsys)
        elif kv.name in ('path', 'pack'):
            fsys_chain.add_sys(fsys)
        else:
            raise ValueError(f'Unknown searchpath key "{kv.real_name}"!')

    sources: Dict[str, PluginSource] = {}

    if hasattr(sys, 'frozen'):
        builtin_transforms = (Path(sys.executable).parent / 'transforms').resolve()
    else:
        # Assume working directory is HammerAddons.
        builtin_transforms = Path('transforms').resolve()

    # Find all the plugins and make plugin objects out of them
    unnamed_ind = 1
    for kv in opts.get(PLUGINS):
        source = PluginSource.parse(kv, expand_path)
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

    return Config(
        opts=opts,
        game=game,
        fsys=fsys_chain,
        pack_blacklist=blacklist,
        plugins=plugin_finder,
        expand_path=expand_path,
    )


def packfile_filters(block: Keyvalues, kind: str) -> Iterator[re_Pattern[str]]:
    """Convert an allowlist/blocklist block into a bunch of regexes."""
    for kv in block:
        if kv.has_children():
            raise ValueError('A keyvalue sub-block is not valid inside the {} filter block!')
        if kv.name in ('path', 'file', 'folder'):
            yield re.compile(re.escape(kv.value.replace('\\', '/')))
        elif kv.name == 'glob':
            # Ensure it matches at the start of the string only.
            yield re.compile('^' + fnmatch.translate(kv.value))
        elif kv.name in ('re', 'regex', 'pattern'):
            yield re.compile(kv.value)
        else:
            raise ValueError(f'Invalid filter type "{kv.real_name}" for {kind}!')


GAMEINFO = Opt.string_or_none(
    'gameinfo',
    """The main game folder. portal2/ for Portal 2, csgo/ for CSGO, etc.
    This is relative to the config file.
    """,
)

AUTO_PACK = Opt.boolean(
    'auto_pack', True,
    """Automatically find and pack files in the map. 
    If this is disabled, specifically-indicated files will still be 
    added as well as their dependencies.
""")

PACK_VPK = Opt.boolean(
    'pack_vpk', False,
    """Allow files in VPKs to be packed into the map. 
    This is disabled by default since these are usually default files.
""")

PACK_DUMP = Opt.string_or_none(
    'pack_dump',
    """If set, copy all the packed resoures to this additional location.
    You can also prefix this with a # character to only copy to this 
    destination, not the BSP pakfile.
""")

PACK_STRIP_CUBEMAPS = Opt.boolean(
    'pack_strip_cubemaps', False,
    f"""If set, strip the generated cubemap files from the BSP. This is necessary for 2013-branch
    games to allow cubemaps to be built properly.
    
    This is equivalent to adding {CUBEMAP_REGEX!r} as a regex "pack_blocklist".
    """
)

PACK_TAGS = Opt.block(
    'pack_tags', Keyvalues('', [Keyvalues(tag, '0') for tag in sorted(USED_PACK_TAGS)]),
    """\
    Specify various tags to indicate what features this game branch includes. This is used
    to accurately include resources for entities that have changed over time.
    """,
)

PACK_ALLOWLIST = Opt.block(
    'pack_allowlist', Keyvalues('', []),
    """\
    Allows forcing specific files or folders to be packed. Each key in this block can be
    either a single file/folder, a glob-style pattern, or an arbitary regex:
    
    * "path" "materials/models/props_expensive/"
    * "path" "scripts/game_sounds_ui.txt"
    * "glob" "*.nut"
    * "regex" "materials/(metal|concrete)/(courtyard|lobby)/*+\\.vmt"
    
    This overrides the blocklist, and also specifications in searchpaths.
    """,
)

PACK_BLOCKLIST = Opt.block(
    'pack_blocklist', Keyvalues('', []),
    """\
    Allows preventing specific files or folders from being packed. The format is the same as 
    'pack_allowlist'. Files generated by the postcompiler itself will always be packed. This will
    be checked against files already present in the BSP, so things like cubemaps can be removed.
    """,
)

SEARCHPATHS = Opt.block(
    'searchpaths', Keyvalues('', []),
    """\
    Specify additional locations to search for files, or configure whether existing locations pack
    or not. Each key-value pair defines a path, with the value either a folder path or a VPK 
    filename relative to the game root. You can also specify specific app ids that will get mounted with the <appid> operator.
    For example: <620>/portal2 will mount the portal2 folder from appid 620; that is Portal 2.
    The key defines the behaviour:
    * "prefix" "folder/" adds the path to the start, so it overrides all others.
    * "path" "vpk_path.vpk" adds the path to the end, so it is checked last.
    * "nopack" "folder/" prohibits files in this path from being packed, you'll need to use one of the others also to add the path.
""")

SOUNDSCRIPT_MANIFEST = Opt.boolean(
    'soundscript_manifest', False,
    """Generate and pack game_sounds_manifest.txt, with all used soundscripts.     
    This is needed to make packing soundscripts work for the Portal 2 
    workshop.
    """,
)

PARTICLES_MANIFEST = Opt.string(
    'particles_manifest', '',
    """If set to a path, generate and pack a particles manifest under this name.     
    This is needed to make packing particles work. "<map name>" is replaced with the map name.
    Depending on your game, these are some of the correct paths:
    * particles/particles_manifest.txt
    * maps/<map name>_particles.txt (TF2, Portal 2)
    * particles/<map name>_manifest.txt (L4D2)
    """,
)

STUDIOMDL = Opt.string(
    'studiomdl', 'bin/studiomdl.exe',
    """Set the path to StudioMDL so the compiler can generate props.
    If blank these features are disabled.
    This is relative to the game root.
    """,
)

MODEL_COMPILE_DUMP = Opt.string(
    'modelcompile_dump', '',
    """If set, models will be compiled as subfolders of this folder, instead of in a 
    temporary directory. The specified folder will be emptied at the start of each compile, to 
    prevent it filling up with old model sources. Move things out that you want to keep.
""")

USE_COMMA_SEP = Opt.boolean_or_none(
    'use_comma_sep',
    """Before L4D, entity I/O used ',' to seperate the different parts.

   Later games used a special symbol to delimit the sections, allowing
   commas to be used in outputs. The compiler will guess which to use
   based on existing outputs in the map, but if this is incorrect 
   (or if there aren't any in the map), use this to override.
""")

PROPCOMBINE_QC_FOLDER = Opt.block(
    'propcombine_qc_folder', Keyvalues('', [Keyvalues('Path', f'|{PATH_KEY_GAME}|../content')]),
    """Define where the QC files are for combinable static props.
    This path is searched recursively. This defaults to 
    the 'content/' folder, which is adjacent to the game root.
    This is how Valve sets up their file structure.
""")

PROPCOMBINE_CROWBAR = Opt.boolean(
    'propcombine_crowbar', True,
    """If enabled, Crowbar will be used to decompile models which don't have
    a QC in the provided QC folder.
""")

PROPCOMBINE_CACHE = Opt.string(
    'propcombine_cache', f"|{PATH_KEY_GAME}|/decomp_cache/",
    """Cache location for models decompiled for combining."""
)

PROPCOMBINE_VOLUME_TOLERANCE = Opt.floating(
    'propcombine_volume_tolerance', -1.0,
    """When propcombining, an attempt will be made to merge collision meshes.
    
    If shrink wrapping a pair of meshes changes the volume less than this,
    the combined version will be used. If negative, this will not be done.
    """
)
PROPCOMBINE_MIN_AUTO_RANGE = Opt.integer(
    'propcombine_auto_range', 0,
    """If greater than zero, combine props at least this close together.""",
)
PROPCOMBINE_MAX_AUTO_RANGE = Opt.integer_or_none(
    'propcombine_max_auto_range',
    """If set, do not automatically combine props further away than this from each other.""",
)

PROPCOMBINE_MIN_CLUSTER = Opt.integer(
    'propcombine_min_cluster', 2,
    """The minimum number of props required before propcombine will
    bother merging them, in propcombine volumes. Should be greater than 1.
    """,
)

PROPCOMBINE_MIN_CLUSTER_AUTO = Opt.integer(
    'propcombine_min_cluster_auto', 0,
    """The minimum number of props required before the automatic propcombine clustering will
    merge the props. If less than or equal to 1, `propcombine_min_cluster` is used.
    """,
)

PROPCOMBINE_BLACKLIST = Opt.block(
    'propcombine_blacklist', Keyvalues('', []),
    """Models specified here will never be propcombined.

    You can specify a full path, or one with * wildcards. Alternatively,
    set 'no_propcombine' in the model $keyvalues.
    """,
)

PROPCOMBINE_PACK = Opt.boolean(
    'propcombine_pack', True,
    """If set, force-pack the combined props."""
)

PLUGINS = Opt.block(
    'plugins', Keyvalues('', []),
    """\
    Add plugins to the post compiler. Each block is a package of plugins in some folder.
    The name must be a Python identifier - the plugins are mounted at 
    "hammeraddons.bsp_transforms.plugin.blockname.filename".
    * "path" must be set to either a single Python file, or a folder of files.
    * If "recurse" is set, subfolders are recursively loaded as packages.
    The transforms folder inside the postcompiler folder is also always
    loaded, under the name "builtin".
""")

TRANSFORM_OPTS = Opt.block(
    'transform_opts', Keyvalues('', []),
    """Specify additional options specific to transforms. Each key here is the name of the 
    transform, and the value is then decided by that transform.
    """
)

DISABLED_TRANSFORMS = Opt.string(
    'transform_disable', '',
    """Specify transforms to disable as a comma-separated string."""
)
