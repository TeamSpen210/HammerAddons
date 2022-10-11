"""Implements "unified" FGD files.

This allows sharing definitions among different engine versions.
"""
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from lzma import LZMAFile
from typing import (
    Union, Optional, TypeVar, Callable,
    Dict, List, Tuple, Set, FrozenSet,
    MutableMapping,
)

from srctools.fgd import (
    FGD, validate_tags, match_tags,
    EntityDef, EntityTypes, IODef,
    KeyValues, ValueTypes,
    Helper, HelperExtAppliesTo, HelperWorldText, HelperSprite, HelperModel,
    AutoVisgroup,
)
from srctools import fgd
from srctools.filesys import RawFileSystem


# Chronological order of games.
# If 'since_hl2' etc is used in FGD, all future games also include it.
# If 'until_l4d' etc is used in FGD, only games before include it.
GAMES_CHRONO: List[Tuple[str, str]] = [
    ('HL2', 'Half-Life 2'),
    ('EP1', 'Half-Life 2 Episode 1'),
    ('EP2', 'Half-Life 2 Episode 2'),

    ('TF2',   'Team Fortress 2'),
    ('P1',    'Portal'),
    ('L4D',   'Left 4 Dead'),
    ('L4D2',  'Left 4 Dead 2'),
    ('ASW',   'Alien Swarm'),
    ('P2',    'Portal 2'),
    ('CSGO',  'Counter-Strike Global Offensive'),

    ('SFM',   'Source Filmmaker'),
    ('DOTA2', 'Dota 2'),
]

# Additional mods/games, which branched off of mainline ones.
MODS_BRANCHED: Dict[str, List[Tuple[str, str]]] = {
    'HL2': [
        ('HLS', 'Half-Life: Source'),
        ('DODS', 'Day of Defeat: Source'),
        ('CSS',  'Counter-Strike: Source'),
    ],
    'EP2': [
        ('MESA', 'Black Mesa'),
        ('GMOD', "Gary's Mod"),
        ('EZ1', 'Entropy Zero'),
        ('EZ2', 'Entropy Zero 2'),
    ],
    'P2': [
        ('P2SIXENSE', 'Portal 2 Sixense MotionPack'),
        ('P2EDU', 'Portal 2 Educational Version'),
        ('STANLEY', 'The Stanley Parable'),
        ('INFRA', 'INFRA'),
    ],
    'CSGO': [
        ('P2DES', 'Portal 2: Desolation'),
    ],
}
MOD_TO_BRANCH = {
    mod: branch
    for branch, mods in MODS_BRANCHED.items()
    for mod, desc in mods
}
ALL_MODS = {
    *MOD_TO_BRANCH,
    'MBASE',  # Mapbase can either be episodic or hl2 base, specify it with those.
}
GAME_ORDER = [game for game, desc in GAMES_CHRONO]
ALL_GAMES = set(GAME_ORDER)

# Specific features that are backported to various games.

FEATURES: Dict[str, Set[str]] = {
    'EP1': {'HL2'},
    'EP2': {'HL2', 'EP1'},

    'MBASE': {'VSCRIPT'},
    'MESA': {'INST_IO'},
    'GMOD': {'HL2', 'EP1', 'EP2'},
    'EZ1': {'MBASE', 'VSCRIPT'},
    'EZ2': {'MBASE', 'VSCRIPT'},

    'L4D2': {'INST_IO', 'VSCRIPT'},
    'TF2': {'PROP_SCALING'},
    'ASW': {'INST_IO', 'VSCRIPT'},
    'P2': {'INST_IO', 'VSCRIPT'},
    'CSGO': {'INST_IO', 'PROP_SCALING', 'VSCRIPT', 'PROPCOMBINE'},
    'P2DES': {'P2'},
}

ALL_FEATURES = {
    tag.upper()
    for t in FEATURES.values()
    for tag in t
}

# Specially handled tags.
TAGS_SPECIAL = {
  'ENGINE',  # Tagged on entries that specify machine-oriented types and defaults.
  'SRCTOOLS',  # Implemented by the srctools post-compiler.
  'PROPPER',  # Propper's added pseudo-entities.
  'COMPLETE',  # KVs that exist, but aren't often or usually used.
  'BEE2',  # BEEmod's templates.
}

ALL_TAGS = {
    *ALL_GAMES, *ALL_MODS, *ALL_FEATURES, *TAGS_SPECIAL,
    *{
        prefix + t.upper()
        for prefix in ['SINCE_', 'UNTIL_']
        for t in GAME_ORDER
    },
}

# If the tag is present, run to backport newer FGD syntax to older engines.
POLYFILLS: List[Tuple[str, Callable[[FGD], None]]] = []
PolyfillFuncT = TypeVar('PolyfillFuncT', bound=Callable[[FGD], None])

# This ends up being the C1 Reverse Line Feed in CP1252,
# which Hammer displays as nothing. We can suffix visgroups with this to
# have duplicates with the same name.
VISGROUP_SUFFIX = '\x8D'

# Special classname which has all the keyvalues and IO of CBaseEntity.
BASE_ENTITY = '_CBaseEntity_'


# Helpers which are only used by one or two entities each.
UNIQUE_HELPERS = {
    fgd.HelperBreakableSurf, fgd.HelperDecal,
    fgd.HelperEnvSprite, fgd.HelperInstance, fgd.HelperLight, fgd.HelperLightSpot,
    fgd.HelperModelLight, fgd.HelperOverlay, fgd.HelperOverlayTransition, fgd.HelperWorldText,
}


def _polyfill(*tags: str) -> Callable[[PolyfillFuncT], PolyfillFuncT]:
    """Register a polyfill, which backports newer FGD syntax to older engines."""
    def deco(func: PolyfillFuncT) -> PolyfillFuncT:
        """Registers the function."""
        for tag in tags:
            POLYFILLS.append((tag.upper(), func))
        if not tags:
            POLYFILLS.append(('', func))
        return func
    return deco


@_polyfill('until_asw', 'mesa')
def _polyfill_boolean(fgd: FGD):
    """Before Alien Swarm's Hammer, boolean was not available as a keyvalue type.

    Substitute with choices.
    """
    for ent in fgd.entities.values():
        for tag_map in ent.keyvalues.values():
            for kv in tag_map.values():
                if kv.type is ValueTypes.BOOL:
                    kv.type = ValueTypes.CHOICES
                    kv.val_list = [
                        ('0', 'No', frozenset()),
                        ('1', 'Yes', frozenset())
                    ]


@_polyfill('until_asw')
def _polyfill_particlesystem(fgd: FGD):
    """Before Alien Swarm's Hammer, the particle system viewer was not available.

    Substitute with just a string.
    """
    for ent in fgd.entities.values():
        for tag_map in ent.keyvalues.values():
            for kv in tag_map.values():
                if kv.type is ValueTypes.STR_PARTICLE:
                    kv.type = ValueTypes.STRING


@_polyfill('until_asw')
def _polyfill_node_id(fgd: FGD):
    """Before Alien Swarm's Hammer, node_id was not available as a keyvalue type.

    Substitute with integer.
    """
    for ent in fgd.entities.values():
        for tag_map in ent.keyvalues.values():
            for kv in tag_map.values():
                if kv.type is ValueTypes.TARG_NODE_SOURCE:
                    kv.type = ValueTypes.INT


@_polyfill('until_l4d2')
def _polyfill_scripts(fgd: FGD):
    """Before L4D2's Hammer, the vscript specific types were not available

    Substitute with just a string.
    """
    for ent in fgd.entities.values():
        for tag_map in ent.keyvalues.values():
            for kv in tag_map.values():
                if kv.type is ValueTypes.STR_VSCRIPT or kv.type is ValueTypes.STR_VSCRIPT_SINGLE:
                    kv.type = ValueTypes.STRING
        for tag_map in ent.inputs.values():
            for inp in tag_map.values():
                if inp.type is ValueTypes.STR_VSCRIPT_SINGLE:
                    inp.type = ValueTypes.STRING


@_polyfill('until_csgo')
def _polyfill_worldtext(fgd: FGD):
    """Strip worldtext(), since this is not available."""
    for ent in fgd:
        ent.helpers[:] = [
            helper
            for helper in ent.helpers
            if not isinstance(helper, HelperWorldText)
        ]


@_polyfill()
def _polyfill_ext_valuetypes(fgd: FGD) -> None:
    # Convert extension types to their real versions.
    decay = {
        ValueTypes.EXT_STR_TEXTURE: ValueTypes.STRING,
        ValueTypes.EXT_ANGLE_PITCH: ValueTypes.FLOAT,
        ValueTypes.EXT_ANGLES_LOCAL: ValueTypes.ANGLES,
        ValueTypes.EXT_VEC_DIRECTION: ValueTypes.VEC,
        ValueTypes.EXT_VEC_LOCAL: ValueTypes.VEC,
    }
    for ent in fgd.entities.values():
        for tag_map in ent.keyvalues.values():
            for kv in tag_map.values():
                kv.type = decay.get(kv.type, kv.type)


def format_all_tags() -> str:
    """Append a formatted description of all allowed tags to a message."""

    return (
        f'- Games: {", ".join(GAME_ORDER)}\n'
        '- SINCE_<game>\n'
        '- UNTIL_<game>\n'
        f' Mods: {", ".join(sorted(ALL_MODS))}\n'
        f'- Features: {", ".join(ALL_FEATURES)}\n'
        f'- Special: {", ".join(TAGS_SPECIAL)}\n'
     )


def expand_tags(tags: FrozenSet[str]) -> FrozenSet[str]:
    """Expand the given tags, producing the full list of tags these will search.

    This adds since_/until_ tags, and values in FEATURES.
    """
    exp_tags = set(tags)
    for tag in tags:
        try:
            exp_tags.add(MOD_TO_BRANCH[tag.upper()])
        except KeyError:
            pass

    for tag in list(exp_tags):
        try:
            exp_tags.update(FEATURES[tag.upper()])
        except KeyError:
            pass
        try:
            pos = GAME_ORDER.index(tag.upper())
        except ValueError:
            pass
        else:
            exp_tags.update(
                'SINCE_' + tag
                for tag in GAME_ORDER[:pos+1]
            )
            exp_tags.update(
                'UNTIL_' + tag
                for tag in GAME_ORDER[pos+1:]
            )
    return frozenset(exp_tags)


def ent_path(ent: EntityDef) -> str:
    """Return the path in the database this entity should be found at."""
    # Very special entity, put in root.
    if ent.classname == 'worldspawn':
        return 'worldspawn.fgd'

    if ent.type is EntityTypes.BASE:
        folder = 'bases'
    else:
        if ent.type is EntityTypes.BRUSH:
            folder = 'brush'
        else:
            folder = 'point'

        if '_' in ent.classname:
            folder += '/' + ent.classname.split('_', 1)[0]

    return '{}/{}.fgd'.format(folder, ent.classname)


def load_database(dbase: Path, extra_loc: Path=None, fgd_vis: bool=False) -> Tuple[FGD, EntityDef]:
    """Load the entire database from disk. This returns the FGD, plus the CBaseEntity definition."""
    print(f'Loading database {dbase}:')
    fgd = FGD()

    fgd.map_size_min = -16384
    fgd.map_size_max = 16384

    # Classname -> filename
    ent_source: Dict[str, str] = {}

    fsys = RawFileSystem(str(dbase))
    for file in dbase.rglob("*.fgd"):
        # Use a temp FGD class, to allow us to verify no overwrites.
        file_fgd = FGD()
        rel_loc = str(file.relative_to(dbase))
        file_fgd.parse_file(
            fsys,
            fsys[rel_loc],
            eval_bases=False,
            encoding='utf8',
        )
        for clsname, ent in file_fgd.entities.items():
            if clsname in fgd.entities:
                raise ValueError(
                    f'Duplicate "{clsname}" class '
                    f'in {rel_loc} and {ent_source[clsname]}!'
                )
            fgd.entities[clsname] = ent
            ent_source[clsname] = rel_loc

        if fgd_vis:
            for parent, visgroup in file_fgd.auto_visgroups.items():
                try:
                    existing_group = fgd.auto_visgroups[parent]
                except KeyError:
                    fgd.auto_visgroups[parent] = visgroup
                else:  # Need to merge
                    existing_group.ents.update(visgroup.ents)

        fgd.mat_exclusions.update(file_fgd.mat_exclusions)
        for tags, mat_list in file_fgd.tagged_mat_exclusions.items():
            fgd.tagged_mat_exclusions[tags] |= mat_list

        print('.', end='', flush=True)

    load_visgroup_conf(fgd, dbase)

    if extra_loc is not None:
        print('\nLoading extra file:')
        if extra_loc.is_file():
            # One file.
            fsys = RawFileSystem(str(extra_loc.parent))
            fgd.parse_file(
                fsys,
                fsys[extra_loc.name],
                eval_bases=False,
            )
        else:
            print('\nLoading extra files:')
            fsys = RawFileSystem(str(extra_loc))
            for file in extra_loc.rglob("*.fgd"):
                fgd.parse_file(
                    fsys,
                    fsys[str(file.relative_to(extra_loc))],
                    eval_bases=False,
                )
                print('.', end='', flush=True)
    print()

    fgd.apply_bases()
    print('\nDone!')

    print('Entities without visgroups:')
    vis_ents = {
        name.casefold()
        for group in fgd.auto_visgroups.values()
        for name in group.ents
    }
    vis_count = ent_count = 0
    for ent in fgd:
        # Base ents, worldspawn, or engine-only ents don't need visgroups.
        if ent.type is EntityTypes.BASE or ent.classname == 'worldspawn':
            continue
        applies_to = get_appliesto(ent)
        if '+ENGINE' in applies_to or 'ENGINE' in applies_to:
            continue
        ent_count += 1
        if ent.classname.casefold() not in vis_ents:
            print(ent.classname, end=', ')
        else:
            vis_count += 1
    print(f'\nVisgroup count: {vis_count}/{ent_count} ({vis_count*100/ent_count:.2f}%) done!')

    try:
        base_entity_def = fgd.entities.pop(BASE_ENTITY.casefold())
        base_entity_def.type = EntityTypes.BASE
    except KeyError:
        base_entity_def = EntityDef(EntityTypes.BASE)
    return fgd, base_entity_def


def load_visgroup_conf(fgd: FGD, dbase: Path) -> None:
    """Parse through the visgroup.cfg file, adding these visgroups."""
    cur_path: List[str] = []
    # Visgroups don't allow duplicating names. Work around that by adding an
    # invisible suffix.
    group_count: Dict[str, int] = Counter()
    try:
        f = (dbase / 'visgroups.cfg').open()
    except FileNotFoundError:
        return
    with f:
        for line in f:
            indent = len(line) - len(line.lstrip('\t'))
            line = line.strip()
            if not line or line.startswith(('#', '//')):
                continue
            cur_path = cur_path[:indent]  # Dedent
            bulleted = line[0] in '-*'
            if (bulleted and '`' not in line) or '(' in line or ')' in line:  # Visgroup.
                single_ent: Optional[str]
                try:
                    vis_name, single_ent = line.lstrip('*-').split('(', 1)
                except ValueError:
                    vis_name = line[1:].strip()
                    single_ent = None
                else:
                    vis_name = vis_name.strip()
                    single_ent = single_ent.strip(' \t`)')

                dupe_count = group_count[vis_name.casefold()]
                if dupe_count:
                    vis_name = vis_name + (VISGROUP_SUFFIX * dupe_count)
                group_count[vis_name.casefold()] = dupe_count + 1

                cur_path.append(vis_name)
                try:
                    visgroup = fgd.auto_visgroups[vis_name.casefold()]
                except KeyError:
                    if indent == 0:  # Don't add Auto itself.
                        continue
                    visgroup = fgd.auto_visgroups[vis_name.casefold()] = AutoVisgroup(vis_name, cur_path[-2])
                if single_ent is not None:
                    visgroup.ents.add(single_ent.casefold())

            elif bulleted:  # Entity.
                ent_name = line[1:].strip('\t `')
                for vis_parent, vis_name in zip(cur_path, cur_path[1:]):
                    visgroup = fgd.auto_visgroups[vis_name.casefold()]
                    visgroup.ents.add(ent_name)


def get_appliesto(ent: EntityDef) -> List[str]:
    """Ensure exactly one AppliesTo() helper is present, and return the args.

    If no helper exists, one will be prepended. Otherwise only the first
    will remain, with the arguments merged together. The same list is
    returned, so it can be viewed or edited.
    """
    pos = None
    applies_to: Set[str] = set()
    for i, helper in enumerate(ent.helpers):
        if isinstance(helper, HelperExtAppliesTo):
            if pos is None:
                pos = i
            applies_to.update(helper.tags)

    if pos is None:
        pos = 0
    arg_list = list(map(str.upper, applies_to))
    arg_list.sort()
    ent.helpers[:] = [
        helper for helper in ent.helpers
        if not isinstance(helper, HelperExtAppliesTo)
    ]
    ent.helpers.insert(pos, HelperExtAppliesTo(arg_list))
    return arg_list


def add_tag(tags: FrozenSet[str], new_tag: str) -> FrozenSet[str]:
    """Modify these tags such that they allow the new tag."""
    is_inverted = new_tag.startswith(('!', '-'))

    # Already allowed/disallowed.
    if match_tags(expand_tags(frozenset({new_tag})), tags) != is_inverted:
        return tags

    tag_set = set(tags)
    if is_inverted:
        tag_set.discard(new_tag[1:])
        tag_set.add(new_tag)
    else:
        tag_set.discard('!' + new_tag.upper())
        tag_set.discard('-' + new_tag.upper())
        if ('+' + new_tag.upper()) not in tag_set:
            tag_set.add(new_tag.upper())

    return frozenset(tag_set)


def check_ent_sprites(ent: EntityDef, used: Dict[str, List[str]]) -> None:
    """Check if the specified entity has a unique sprite."""
    mdl: Optional[str] = None
    sprite: Optional[str] = None
    for helper in ent.helpers:
        if type(helper) in UNIQUE_HELPERS:
            return  # Specialised helper is sufficient.
        if isinstance(helper, fgd.HelperModel):
            if helper.model is None and 'model' in ent.kv:
                return  # Model is customisable.
            mdl = helper.model
        if isinstance(helper, fgd.HelperSprite):
            if helper.mat is None:
                print(f'{ent.classname}: {helper}???')
            sprite = helper.mat
    # If both model and sprite, allow model to be duplicate.
    if mdl and sprite:
        display = sprite
    elif mdl:
        display = mdl
    elif sprite:
        display = sprite
    else:
        if '+ENGINE' not in get_appliesto(ent):
            print(f'{ent.classname}: No sprite/model? {", ".join(map(repr, ent.helpers))}')
        return

    display = display.casefold()
    if display in used:
        print(f'{ent.classname}: Reuses {display}: {used[display]}')
    used[display].append(ent.classname)


def action_count(
    dbase: Path,
    extra_db: Optional[Path],
    factories_folder: Path,
    plot: bool=False,
) -> None:
    """Output a count of all entities in the database per game."""
    fgd, base_entity_def = load_database(dbase, extra_db)

    count_base: Dict[str, int] = Counter()
    count_point: Dict[str, int] = Counter()
    count_brush: Dict[str, int] = Counter()

    all_tags = set()

    for ent in fgd:
        for tag in get_appliesto(ent):
            all_tags.add(tag.lstrip('+-!').upper())

    games = (ALL_GAMES | ALL_MODS) & all_tags

    print('Done.\nGames: ' + ', '.join(sorted(games)))

    expanded: Dict[str, FrozenSet[str]] = {
        game: expand_tags(frozenset({game}))
        for game in ALL_GAMES | ALL_MODS
    }
    expanded['ALL'] = frozenset()

    game_classes: MutableMapping[Tuple[str, str], Set[str]] = defaultdict(set)
    base_uses: MutableMapping[str, Set[str]] = defaultdict(set)
    all_ents: MutableMapping[str, Set[str]] = defaultdict(set)

    for ent in fgd:
        if ent.type is EntityTypes.BASE:
            counter = count_base
            typ = 'Base'
            # Ensure it's present, so we detect 0-use bases.
            base_uses[ent.classname]  # noqa
        elif ent.type is EntityTypes.BRUSH:
            counter = count_brush
            typ = 'Brush'
        else:
            counter = count_point
            typ = 'Point'
        appliesto = get_appliesto(ent)

        has_ent = set()

        for base in ent.bases:
            assert isinstance(base, EntityDef)
            base_uses[base.classname].add(ent.classname)

        for game, tags in expanded.items():
            if match_tags(tags, appliesto):
                counter[game] += 1
                game_classes[game, typ].add(ent.classname)
                has_ent.add(game)
            # Allow explicitly saying certain ents aren't in the actual game
            # with the "engine" tag, or only adding them to this + the binary dump.
            if ent.type is not EntityTypes.BASE and match_tags(tags | {'ENGINE'}, appliesto):
                all_ents[game].add(ent.classname.casefold())

        has_ent.discard('ALL')

        if has_ent == games:
            # Applies to all, strip.
            game_classes['ALL', typ].add(ent.classname)
            counter['ALL'] += 1
            if appliesto:
                print('ALL game: ', ent.classname)
            for game in games:
                counter[game] -= 1
                game_classes[game, typ].discard(ent.classname)

    all_games: Set[str] = {*count_base, *count_point, *count_brush}

    def ordering(game: str) -> tuple:
        """Put ALL at the start, mods at the end."""
        if game == 'ALL':
            return (0, 0)
        try:
            return (1, GAME_ORDER.index(game))
        except ValueError:
            return (2, game)  # Mods

    game_order = sorted(all_games, key=ordering)

    row_temp = '{:^9} | {:^6} | {:^6} | {:^6}'
    header = row_temp.format('Game', 'Base', 'Point', 'Brush')

    print(header)
    print('-' * len(header))

    for game in game_order:
        print(row_temp.format(
            game,
            count_base[game],
            count_point[game],
            count_brush[game],
        ))

    # If matplotlib is installed, render this as a nice graph.
    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None
        else:
            disp_games = game_order[::-1]
            point_count_list = [count_point[game] for game in disp_games]
            solid_count_list = [count_brush[game] for game in disp_games]
            plt.figure(0)
            plt.barh(disp_games, point_count_list)
            plt.barh(disp_games, solid_count_list)
            plt.legend(["Point", "Brush"])
            plt.xticks(range(0, 500, 50))
            plt.show()

    print('\n\nBases:')
    for base, count in sorted(base_uses.items(), key=lambda x: (len(x[1]), x[0])):
        ent = fgd[base]
        if ent.type is EntityTypes.BASE and (
            ent.keyvalues or ent.outputs or ent.inputs
        ):
            print(base, len(count), count if len(count) == 1 else '...')

    print('\n\nEntity Dumps:')
    for dump_path in factories_folder.glob('*.txt'):
        with dump_path.open() as f:
            dump_classes = {
                cls.casefold().strip()
                for cls in f
                if not cls.isspace()
            }
        game = dump_path.stem.upper()
        tags = frozenset(game.split('_'))

        defined_classes = {
            cls
            for tag in tags
            for cls in all_ents.get(tag, ())
            if not cls.startswith('comp_')
        }
        if not defined_classes:
            print(f'No dump for tags "{game}"!')
            continue

        extra = defined_classes - dump_classes
        missing = dump_classes - defined_classes
        if extra:
            print(f'{game} - Extraneous definitions: ')
            print(', '.join(sorted(extra)))
        if missing:
            print(f'{game} - Missing definitions: ')
            print(', '.join(sorted(missing)))

    print('\n\nMissing Class Resources:')

    missing_count = 0
    defined_count = 0
    not_in_engine = {'-ENGINE', '!ENGINE', 'SRCTOOLS', '+SRCTOOLS'}
    for clsname in sorted(fgd.entities):
        ent = fgd.entities[clsname]
        if ent.type is EntityTypes.BASE:
            continue

        if not not_in_engine.isdisjoint(get_appliesto(ent)):
            continue
        if isinstance(ent.resources, tuple):
            print(clsname, end=', ')
            missing_count += 1
        else:
            defined_count += 1

    print(
        f'\nMissing: {missing_count}, '
        f'Defined: {defined_count} = {defined_count/(missing_count + defined_count):.2%}\n\n'
    )

    mdl_or_sprite = defaultdict(list)
    for ent in fgd:
        if ent.type is not EntityTypes.BASE and ent.type is not EntityTypes.BRUSH:
            check_ent_sprites(ent, mdl_or_sprite)


def action_import(
    dbase: Path,
    engine_tag: str,
    fgd_paths: List[Path],
) -> None:
    """Import an FGD file, adding differences to the unified files."""
    new_fgd = FGD()
    print(f'Using tag "{engine_tag}"')

    expanded = expand_tags(frozenset({engine_tag}))

    print(f'Reading FGDs:')
    for path in fgd_paths:
        print(path)
        with RawFileSystem(str(path.parent)) as fsys:
            new_fgd.parse_file(fsys, fsys[path.name], eval_bases=False)

    print(f'\nImporting {len(new_fgd)} entiti{"y" if len(new_fgd) == 1 else "ies"}...')
    for new_ent in new_fgd:
        path = dbase / ent_path(new_ent)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            old_fgd = FGD()
            with RawFileSystem(str(path.parent)) as fsys:
                old_fgd.parse_file(fsys, fsys[path.name], eval_bases=False)
            try:
                ent = old_fgd[new_ent.classname]
            except KeyError:
                raise ValueError("Classname not present in FGD!")
            # Now merge the two.

            if new_ent.desc not in ent.desc:
                # Temporary, append it.
                ent.desc += '|||' + new_ent.desc

            # Merge helpers. We just combine overall...
            for new_base in new_ent.bases:
                if new_base not in ent.bases:
                    ent.bases.append(new_base)

            for helper in new_ent.helpers:
                # Sorta ew, quadratic search. But helper sizes shouldn't
                # get too big.
                if helper not in ent.helpers:
                    ent.helpers.append(helper)

            for cat in ('keyvalues', 'inputs', 'outputs'):
                cur_map: Dict[str, Dict[FrozenSet[str], Union[KeyValues, IODef]]] = getattr(ent, cat)
                new_map = getattr(new_ent, cat)
                new_names = set()
                for name, tag_map in new_map.items():
                    new_names.add(name)
                    try:
                        orig_tag_map = cur_map[name]
                    except KeyError:
                        # Not present in the old file.
                        cur_map[name] = {
                            add_tag(tag, engine_tag): value
                            for tag, value in tag_map.items()
                        }
                        continue
                    # Otherwise merge, if unequal add the new ones.
                    # TODO: Handle tags in "new" files.
                    for tag, new_value in tag_map.items():
                        for old_tag, old_value in orig_tag_map.items():
                            if old_value == new_value:
                                if tag:
                                    # Already present, modify this tag.
                                    del orig_tag_map[old_tag]
                                    orig_tag_map[add_tag(old_tag, engine_tag)] = new_value
                                # else: Blank tag, keep blank.
                                break
                        else:
                            # Otherwise, we need to add this.
                            orig_tag_map[add_tag(tag, engine_tag)] = new_value

                # Make sure removed items don't apply to the new tag.
                for name, tag_map in cur_map.items():
                    if name not in new_names:
                        cur_map[name] = {
                            add_tag(tag, '!' + engine_tag): value
                            for tag, value in tag_map.items()
                        }

        else:
            # No existing one, just set appliesto.
            ent = new_ent

        applies_to = get_appliesto(ent)
        if not match_tags(expanded, applies_to):
            applies_to.append(engine_tag)
        ent.helpers[:] = [
            helper for helper in ent.helpers
            if not isinstance(helper, HelperExtAppliesTo)
        ]

        with open(path, 'w') as f:
            ent.export(f)

        print('.', end='', flush=True)
    print()


def action_export(
    dbase: Path,
    extra_db: Optional[Path],
    tags: FrozenSet[str],
    output_path: Path,
    as_binary: bool,
    engine_mode: bool,
) -> None:
    """Create an FGD file using the given tags."""

    if engine_mode:
        tags = frozenset({'ENGINE'})
    else:
        tags = expand_tags(tags)

    print('Tags expanded to: {}'.format(', '.join(tags)))

    fgd, base_entity_def = load_database(dbase, extra_db)

    if engine_mode:
        # In engine mode, we don't care about specific games.
        print('Collapsing bases...')
        fgd.collapse_bases()

        # Cache these constant sets.
        tags_empty = frozenset('')
        tags_not_engine = frozenset({'-ENGINE', '!ENGINE'})

        print('Merging tags...')
        for ent in list(fgd):
            # If it's set as not in engine, strip.
            if not tags_not_engine.isdisjoint(get_appliesto(ent)):
                del fgd.entities[ent.classname.casefold()]
            # Strip applies-to helper and ordering helper.
            ent.helpers[:] = [
                helper for helper in ent.helpers
                if not helper.IS_EXTENSION
            ]
            # Force everything to inherit from CBaseEntity, since
            # we're then removing any KVs that are present on that.
            if ent.classname != BASE_ENTITY:
                ent.bases = [base_entity_def]

            value: Union[IODef, KeyValues]
            category: Dict[str, Dict[FrozenSet[str], Union[IODef, KeyValues]]]
            base_cat: Dict[str, Dict[FrozenSet[str], Union[IODef, KeyValues]]]
            for attr_name in ['inputs', 'outputs', 'keyvalues']:
                category = getattr(ent, attr_name)
                base_cat = getattr(base_entity_def, attr_name)
                # For each category, check for what value we want to keep.
                # If only one, we keep that.
                # If there's an "ENGINE" tag, that's specifically for us.
                # Otherwise, warn if there's a type conflict.
                # If the final value is choices, warn too (not really a type).
                for key, orig_tag_map in list(category.items()):
                    # Remake the map, excluding non-engine tags.
                    # If any are explicitly matching us, just use that
                    # directly.
                    tag_map: Dict[FrozenSet[str], Union[IODef, KeyValues]] = {}
                    for tags, value in orig_tag_map.items():
                        if 'ENGINE' in tags or '+ENGINE' in tags:
                            if value.type is ValueTypes.CHOICES:
                                raise ValueError(
                                    '{}.{}: Engine tags cannot be '
                                    'CHOICES!'.format(ent.classname, key)
                                )
                            # Use just this.
                            tag_map = {tags_empty: value}
                            break
                        elif '-ENGINE' not in tags and '!ENGINE' not in tags:
                            tag_map[tags] = value

                    if not tag_map:
                        # All were set as non-engine, so it's not present.
                        del category[key]
                        continue
                    elif len(tag_map) == 1:
                        # Only one type, that's the one for the engine.
                        [value] = tag_map.values()
                    else:
                        # More than one tag.
                        # IODef and KeyValues have a type attr.
                        types = {val.type for val in tag_map.values()}
                        if len(types) > 1:
                            print('{}.{} has multiple types! ({})'.format(
                                ent.classname,
                                key,
                                ', '.join([typ.value for typ in types])
                            ))
                        # Pick the one with shortest tags arbitrarily.
                        _, value = min(
                            tag_map.items(),
                            key=lambda t: len(t[0]),
                        )

                    # If it's CHOICES, we can't know what type it is.
                    # Guess either int or string, if we can convert.
                    if value.type is ValueTypes.CHOICES:
                        print(
                            '{}.{} uses CHOICES type, '
                            'provide ENGINE '
                            'tag!'.format(ent.classname, key)
                        )
                        if isinstance(value, KeyValues):
                            assert value.val_list is not None
                            try:
                                for choice_val, name, tag in value.choices_list:
                                    int(choice_val)
                            except ValueError:
                                # Not all are ints, it's a string.
                                value.type = ValueTypes.STRING
                            else:
                                value.type = ValueTypes.INT
                            value.val_list = None

                    # Check if this is a shared property among all ents,
                    # and if so skip exporting.
                    if ent.classname != BASE_ENTITY:
                        base_value: Union[KeyValues, IODef]
                        try:
                            [base_value] = base_cat[key].values()
                        except KeyError:
                            pass
                        except ValueError:
                            raise ValueError(
                                f'Base Entity {attr_name[:-1]} "{key}" '
                                f'has multiple tags: {list(base_cat[key].keys())}'
                            )
                        else:
                            if base_value.type is ValueTypes.CHOICES:
                                print(
                                    f'Base Entity {attr_name[:-1]} '
                                    f'"{key}"  is a choices type!'
                                )
                            elif base_value.type is value.type:
                                del category[key]
                                continue
                            elif attr_name == 'keyvalues' and key == 'model':
                                # This can be sprite or model.
                                pass
                            elif base_value.type is ValueTypes.FLOAT and value.type is ValueTypes.INT:
                                # Just constraining it down to a whole number.
                                pass
                            else:
                                print(f'{ent.classname}.{key}: {value.type} != base {base_value.type}')

                    # Blank this, it's not that useful.
                    value.desc = ''
                    category[key] = {tags_empty: value}

        # Add in the base entity definition, and clear it out.
        fgd.entities[BASE_ENTITY.casefold()] = base_entity_def
        base_entity_def.desc = ''
        base_entity_def.helpers = []
        # Strip out all the tags.
        for cat in [base_entity_def.inputs, base_entity_def.outputs, base_entity_def.keyvalues]:
            for key, tag_map in cat.items():
                [value] = tag_map.values()
                cat[key] = {tags_empty: value}
                if value.type is ValueTypes.CHOICES:
                    raise ValueError('Choices key in CBaseEntity!')
    else:
        print('Culling incompatible entities...')

        ents = list(fgd.entities.values())
        fgd.entities.clear()

        for ent in ents:
            applies_to = get_appliesto(ent)
            if match_tags(tags, applies_to):
                fgd.entities[ent.classname] = ent
                ent.strip_tags(tags)

            # Remove bases that don't apply.
            for base in ent.bases[:]:
                if not match_tags(tags, get_appliesto(base)):
                    ent.bases.remove(base)

    if not engine_mode:
        for poly_tag, polyfill in POLYFILLS:
            if not poly_tag or poly_tag in tags:
                polyfill(fgd)

    print('Applying helpers to child entities and optimising...')
    for ent in fgd.entities.values():
        # Merge them together.
        helpers: List[Helper] = []
        for base in ent.bases:
            helpers.extend(base.helpers)
        helpers.extend(ent.helpers)

        # Then optimise this list.
        ent.helpers.clear()
        for helper in helpers:
            if helper in ent.helpers:  # No duplicates
                continue
            # Strip applies-to helper.
            if isinstance(helper, HelperExtAppliesTo):
                continue

            # For each, check if it makes earlier ones obsolete.
            overrides = helper.overrides()
            if overrides:
                ent.helpers[:] = [
                    helper for helper in ent.helpers
                    if helper.TYPE not in overrides
                ]

            # But it itself should be added to the end regardless.
            ent.helpers.append(helper)

    print('Culling unused bases...')
    used_bases: Set[EntityDef] = set()
    # We only want to keep bases that provide keyvalues. We've merged the
    # helpers in.
    for ent in fgd.entities.values():
        if ent.type is not EntityTypes.BASE:
            for base in ent.iter_bases():
                if base.type is EntityTypes.BASE and (
                    base.keyvalues or base.inputs or base.outputs
                ):
                    used_bases.add(base)

    for classname, ent in list(fgd.entities.items()):
        if ent.type is EntityTypes.BASE:
            if ent not in used_bases:
                del fgd.entities[classname]
                continue
            else:
                # Helpers aren't inherited, so this isn't useful anymore.
                ent.helpers.clear()
        # Cull all base classes we don't use.
        # Ents that inherit from each other always need to exist.
        # We also need to replace bases with their parent, if culled.
        todo = ent.bases.copy()
        done = set(todo)
        ent.bases.clear()
        for base in todo:
            if base.type is not EntityTypes.BASE or base in used_bases:
                ent.bases.append(base)
            else:
                for subbase in base.bases:
                    if subbase not in done:
                        todo.append(subbase)

    print('Merging in material exclusions...')
    for mat_tags, materials in fgd.tagged_mat_exclusions.items():
        if match_tags(tags, mat_tags):
            fgd.mat_exclusions |= materials
    fgd.tagged_mat_exclusions.clear()

    print('Culling visgroups...')
    # Cull visgroups that no longer exist for us.
    valid_ents = {
        ent.classname.casefold()
        for ent in fgd.entities.values()
        if ent.type is not EntityTypes.BASE
    }
    for key, visgroup in list(fgd.auto_visgroups.items()):
        visgroup.ents.intersection_update(valid_ents)
        if not visgroup.ents:
            del fgd.auto_visgroups[key]

    if engine_mode:
        res_tags = defaultdict(set)
        for ent in fgd.entities.values():
            for res in ent.resources:
                for tag in res.tags:
                    res_tags[tag.lstrip('-+!').upper()].add(ent.classname)
        print('Resource tags:')
        for tag, classnames in res_tags.items():
            print(f'- {tag}: {len(classnames)} ents')

    print('Exporting...')

    if as_binary:
        with open(output_path, 'wb') as bin_f, LZMAFile(bin_f, 'w') as comp:
            # Private, reserved for us.
            # noinspection PyProtectedMember
            from srctools._engine_db import serialise
            serialise(fgd, comp)
    else:
        with open(output_path, 'w', encoding='iso-8859-1') as txt_f:
            fgd.export(txt_f)
            # BEE2 compatibility, don't make it run.
            if 'P2' in tags:
                txt_f.write('\n// BEE 2 EDIT FLAG = 0 \n')


def action_visgroup(
    dbase: Path,
    extra_loc: Optional[Path],
    dest: Path,
) -> None:
    """Dump all auto-visgroups into the specified file, using a custom format."""
    fgd, base_entity_def = load_database(dbase, extra_loc, fgd_vis=True)

    # TODO: This shouldn't be copied from fgd.export(), need to make the
    #  parenting invariant guaranteed by the classes.
    vis_by_parent: Dict[str, Set[AutoVisgroup]] = defaultdict(set)

    for visgroup in list(fgd.auto_visgroups.values()):
        if not visgroup.parent:
            visgroup.parent = 'Auto'
        elif visgroup.parent.casefold() not in fgd.auto_visgroups:
            # This is an "orphan" visgroup, not linked back to Auto.
            # Connect it back there, by generating the parent.
            parent_group = fgd.auto_visgroups[visgroup.parent.casefold()] = AutoVisgroup(visgroup.parent, 'Auto')
            parent_group.ents.update(visgroup.ents)
        vis_by_parent[visgroup.parent.casefold()].add(visgroup)

    def write_vis(group: AutoVisgroup, indent: str) -> None:
        """Write a visgroup and its children."""
        children = sorted(vis_by_parent[group.name.casefold()], key=lambda g: g.name)
        # Special case for singleton visgroups - no children, only 1 ent.
        if not children and len(group.ents) == 1:
            [single_ent] = group.ents
            f.write('{}- {} (`{}`)\n'.format(indent, group.name, single_ent))
            return

        # First, write the child visgroups.
        child_indent = indent + '\t'
        f.write('{}- {}\n'.format(indent, group.name))
        for child_group in children:
            write_vis(child_group, child_indent)
        # Then the actual children.
        for child in sorted(group.ents):
            # Visgroups are also in the list.
            if child in fgd.auto_visgroups:
                continue
            # For ents in subfolders, each parent group also lists
            # them. So we want to add it to the group who's children
            # do not contain the ent.
            if all(child not in group.ents for group in children):
                f.write('{}* `{}`\n'.format(child_indent, child))

    print('Writing...')
    with dest.open('w') as f:
        write_vis(AutoVisgroup('Auto', ''), '')


def main(args: List[str]=None):
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Manage a set of unified FGDs, sharing configs between engine versions.",
    )
    # Find the repository root.
    repo_dir = Path(__file__).parents[2]

    parser.add_argument(
        "-d", "--database",
        default=str(repo_dir / "fgd"),
        help="The folder to write the FGD files to or from."
    )
    parser.add_argument(
        "--extra",
        dest="extra_db",
        default=None,
        help="If specified, an additional folder to read FGD files from. "
             "These override the normal database.",
    )
    subparsers = parser.add_subparsers(dest="mode")

    parser_count = subparsers.add_parser(
        "count",
        help=action_count.__doc__,
        aliases=["c"],
    )
    parser_count.add_argument(
        "--plot",
        action="store_true",
        help="Use matplotlib to produce a graph of how many entities are "
             "present in each engine branch.",
    )

    parser_exp = subparsers.add_parser(
        "export",
        help=action_export.__doc__,
        aliases=["exp", "i"],
    )

    parser_exp.add_argument(
        "-o", "--output",
        default="output.fgd",
        help="Destination FGD filename."
    )
    parser_exp.add_argument(
        "-e", "--engine",
        action="store_true",
        help="If set, produce FGD for parsing by script. "
             "This includes all keyvalues regardless of tags, "
             "to allow parsing VMF/BSP files. Overrides tags if "
             " provided.",
    )
    parser_exp.add_argument(
        "-b", "--binary",
        action="store_true",
        help="If set, produce a binary format used by Srctools.",
    )
    parser_exp.add_argument(
        "tags",
        nargs="*",
        help="Tags to include in the output.",
        default=None,
    )

    parser_imp = subparsers.add_parser(
        "import",
        help=action_import.__doc__,
        aliases=["imp", "i"],
    )
    parser_imp.add_argument(
        "engine",
        type=str.upper,
        choices=GAME_ORDER,
        help="Engine to mark this FGD set as supported by.",
    )
    parser_imp.add_argument(
        "fgd",
        nargs="+",
        type=Path,
        help="The FGD files to import. "
    )

    parser_vis = subparsers.add_parser(
        "visgroup",
        help=action_visgroup.__doc__,
        aliases=["vis", "v"],
    )

    parser_vis.add_argument(
        "-o", "--output",
        default="visgroups.md",
        type=Path,
        help="Visgroup dump filename.",
    )

    result = parser.parse_args(args)

    if result.mode is None:
        parser.print_help()
        return

    dbase = Path(result.database).resolve()
    dbase.mkdir(parents=True, exist_ok=True)

    extra_db: Optional[Path]
    if result.extra_db is not None:
        extra_db = Path(result.extra_db).resolve()
    else:
        extra_db = None

    if result.mode in ("import", "imp", "i"):
        action_import(
            dbase,
            result.engine,
            result.fgd,
        )
    elif result.mode in ("export", "exp", "e"):
        # Engine means tags are ignored.
        # Non-engine means tags must be specified!
        if result.engine:
            if result.tags:
                print("Tags ignored in --engine mode...", file=sys.stderr)
            result.tags = ['ENGINE']
        elif not result.tags:
            parser.error("At least one tag must be specified!")

        tags = validate_tags(result.tags)

        for tag in tags:
            if tag not in ALL_TAGS:
                parser.error(f'Invalid tag "{tag}"! Allowed tags: \n{format_all_tags()}')
        action_export(
            dbase,
            extra_db,
            tags,
            result.output,
            result.binary,
            result.engine,
        )
    elif result.mode in ("c", "count"):
        action_count(dbase, extra_db, factories_folder=Path(repo_dir, 'db', 'factories'))
    elif result.mode in ("visgroup", "v", "vis"):
        action_visgroup(dbase, extra_db, result.output)
    else:
        raise AssertionError("Unknown mode! (" + result.mode + ")")


if __name__ == '__main__':
    main(sys.argv[1:])

    #for game in GAME_ORDER:
    #    print('\n'+ game + ':')
    #    main(['export', '-o', 'fgd_out/' + game + '.fgd', game])
