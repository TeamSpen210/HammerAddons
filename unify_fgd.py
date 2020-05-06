"""Implements "unified" FGD files.

This allows sharing definitions among different engine versions.
"""
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from lzma import LZMAFile
from typing import (
    Union, Optional,
    TypeVar,
    Callable,
    Dict, List, Tuple,
    Set, FrozenSet,
    MutableMapping,
)

from srctools.fgd import (
    FGD, validate_tags, match_tags,
    EntityDef, EntityTypes,
    HelperTypes, IODef,
    KeyValues, ValueTypes,
    HelperExtAppliesTo,
    HelperWorldText,
    AutoVisgroup,
)
from srctools.filesys import RawFileSystem


# Chronological order of games.
# If 'since_hl2' etc is used in FGD, all future games also include it.
# If 'until_l4d' etc is used in FGD, only games before include it.

GAMES = [
    ('HLS',  'Half-Life: Source'),
    ('DODS', 'Day of Defeat: Source'),
    ('CSS',  'Counter-Strike: Source'),

    ('HL2',  'Half-Life 2'),
    ('EP1',  'Half-Life 2 Episode 1'),
    ('EP2',  'Half-Life 2 Episode 2'),

    # Not chronologically here, but it uses 2013 as the base.
    ('MBASE', 'Mapbase'),
    # Mesa also appears to be about here...
    ('MESA', 'Black Mesa'),
    ('GMOD', "Gary's Mod"),

    ('TF2',  'Team Fortress 2'),
    ('P1', 'Portal'),
    ('L4D', 'Left 4 Dead'),
    ('L4D2', 'Left 4 Dead 2'),
    ('ASW', 'Alien Swam'),
    ('P2', 'Portal 2'),
    ('INFRA', 'INFRA'),
    ('CSGO', 'Counter-Strike Global Offensive'),

    ('SFM', 'Source Filmmaker'),
    ('DOTA2', 'Dota 2'),
    ('PUNT', 'PUNT'),
    ('P2DES', 'Portal 2: Desolation'),
]  # type: List[Tuple[str, str]]

GAME_ORDER = [game for game, desc in GAMES]
GAME_NAME = dict(GAMES)

# Specific features that are backported to various games.

FEATURES: Dict[str, Set[str]] = {
    # 2013 engine backports this.
    'HL2': {'INSTANCING'},
    'EP1': {'INSTANCING'},
    'EP2': {'INSTANCING'},

    'MBASE': {'INSTANCING'},
    'MESA': {'INSTANCING', 'INST_IO'},
    'GMOD': {'HL2', 'EP1', 'EP2'},
    
    'L4D': {'INSTANCING'},
    'L4D2': {'INSTANCING', 'INST_IO', 'VSCRIPT'},
    'TF2': {'INSTANCING', 'PROP_SCALING'},
    'ASW': {'INSTANCING', 'INST_IO', 'VSCRIPT'},
    'P2': {'INSTANCING', 'INST_IO', 'VSCRIPT'},
    'CSGO': {'INSTANCING', 'INST_IO', 'PROP_SCALING', 'VSCRIPT', 'PROPCOMBINE'},
    'INFRA': {'P2', 'INSTANCING', 'INST_IO', 'VSCRIPT'},
    'P2DES': {'P2', 'INSTANCING', 'INST_IO', 'PROP_SCALING', 'VSCRIPT', 'PROPCOMBINE'},
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
  'BEE2',  # BEEmod's templates.
}

ALL_TAGS = set()  # type: Set[str]
ALL_TAGS.update(GAME_ORDER)
ALL_TAGS.update(ALL_FEATURES)
ALL_TAGS.update(TAGS_SPECIAL)
ALL_TAGS.update('SINCE_' + t.upper() for t in GAME_ORDER)
ALL_TAGS.update('UNTIL_' + t.upper() for t in GAME_ORDER)


# If the tag is present, run to backport newer FGD syntax to older engines.
POLYFILLS = []  # type: List[Tuple[str, Callable[[FGD], None]]]


PolyfillFuncT = TypeVar('PolyfillFuncT', bound=Callable[[FGD], None])

# This ends up being the C1 Reverse Line Feed in CP1252,
# which Hammer displays as nothing. We can suffix visgroups with this to
# have duplicates with the same name.
VISGROUP_SUFFIX = '\x8D'


def _polyfill(*tags: str) -> Callable[[PolyfillFuncT], PolyfillFuncT]:
    """Register a polyfill, which backports newer FGD syntax to older engines."""
    def deco(func: PolyfillFuncT) -> PolyfillFuncT:
        for tag in tags:
            POLYFILLS.append((tag.upper(), func))
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


@_polyfill('until_csgo')
def _polyfill_worltext(fgd: FGD):
    """Strip worldtext(), since this is not available."""
    for ent in fgd:
        ent.helpers[:] = [
            helper
            for helper in ent.helpers
            if not isinstance(helper, HelperWorldText)
        ]


def format_all_tags() -> str:
    """Append a formatted description of all allowed tags to a message."""
    
    return (
        '- Games: {}\n'
        '- SINCE_<game>\n'
        '- UNTIL_<game>\n'
        '- Features: {}\n'
        '- Special: {}\n'
     ).format(
         ', '.join(GAME_ORDER),
         ', '.join(ALL_FEATURES),
        ', '.join(TAGS_SPECIAL),
     )


def expand_tags(tags: FrozenSet[str]) -> FrozenSet[str]:
    """Expand the given tags, producing the full list of tags these will search.

    This adds since_/until_ tags, and values in FEATURES.
    """
    exp_tags = set(tags)
    for tag in tags:
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


def load_database(dbase: Path, extra_loc: Path=None, fgd_vis: bool=False) -> FGD:
    """Load the entire database from disk."""
    print('Loading database:')
    fgd = FGD()

    fgd.map_size_min = -16384
    fgd.map_size_max = 16384

    # Classname -> filename
    ent_source: Dict[str, str] = {}

    with RawFileSystem(str(dbase)) as fsys:
        for file in dbase.rglob("*.fgd"):
            # Use a temp FGD class, to allow us to verify no overwrites.
            file_fgd = FGD()
            rel_loc = str(file.relative_to(dbase))
            file_fgd.parse_file(
                fsys,
                fsys[rel_loc],
                eval_bases=False,
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

            print('.', end='', flush=True)

    load_visgroup_conf(fgd, dbase)

    if extra_loc is not None:
        print('\nLoading extra file:')
        if extra_loc.is_file():
            # One file.
            with RawFileSystem(str(extra_loc.parent)) as fsys:
                fgd.parse_file(
                    fsys,
                    fsys[extra_loc.name],
                    eval_bases=False,
                )
        else:
            print('\nLoading extra files:')
            with RawFileSystem(str(extra_loc)) as fsys:
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

    return fgd


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
            if not line:
                continue
            cur_path = cur_path[:indent]  # Dedent
            if line.startswith('-') or '(' in line or ')' in line:  # Visgroup.
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

            elif line.startswith('*'):  # Entity.
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
        help for help in ent.helpers
        if not isinstance(help, HelperExtAppliesTo)
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


def action_count(dbase: Path, extra_db: Optional[Path], plot: bool=False) -> None:
    """Output a count of all entities in the database per game."""
    fgd = load_database(dbase, extra_db)

    count_base: Dict[str, int] = Counter()
    count_point: Dict[str, int] = Counter()
    count_brush: Dict[str, int] = Counter()

    all_tags = set()

    for ent in fgd:
        for tag in get_appliesto(ent):
            all_tags.add(tag.lstrip('+-!').upper())

    games = set(GAME_ORDER).intersection(all_tags)

    print('Done.\nGames: ' + ', '.join(sorted(games)))

    expanded: Dict[str, FrozenSet[str]] = {
        game: expand_tags(frozenset({game}))
        for game in GAME_ORDER
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

    game_order = ['ALL'] + sorted(all_games - {'ALL'}, key=GAME_ORDER.index)

    row_temp = '{:<5} | {:^6} | {:^6} | {:^6}'
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
    for dump_path in Path('db', 'factories').glob('*.txt'):
        with dump_path.open() as f:
            dump_classes = {
                cls.casefold().strip()
                for cls in f
                if not cls.isspace()
            }
        game = dump_path.stem.upper()
        try:
            defined_classes = all_ents[game]
        except KeyError:
            print(f'No dump for tag "{game}"!')
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
    from srctools.packlist import CLASS_RESOURCES

    missing_count = 0
    for clsname in sorted(fgd.entities):
        ent = fgd.entities[clsname]
        if ent.type is EntityTypes.BASE:
            continue

        applies_to = get_appliesto(ent)
        if '-ENGINE' in applies_to or '!ENGINE' in applies_to:
            continue

        if clsname not in CLASS_RESOURCES:
            print(clsname, end=', ')
            missing_count += 1
    print('\nMissing:', missing_count)

    print('Extra ents: ')
    for clsname in CLASS_RESOURCES:
        if clsname not in fgd.entities:
            print(clsname, end=', ')
    print('\n')


def action_import(
    dbase: Path,
    engine_tag: str,
    fgd_paths: List[Path],
) -> None:
    """Import an FGD file, adding differences to the unified files."""
    new_fgd = FGD()
    print('Using tag "{}"'.format(engine_tag))

    expanded = expand_tags(frozenset({engine_tag}))

    print('Reading FGDs:'.format(len(fgd_paths)))
    for path in fgd_paths:
        print(path)
        with RawFileSystem(str(path.parent)) as fsys:
            new_fgd.parse_file(fsys, fsys[path.name], eval_bases=False)

    print('\nImporting {} entiti{}...'.format(
        len(new_fgd),
        "y" if len(new_fgd) == 1 else "ies",
    ))
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
                cur_map = getattr(ent, cat)  # type: Dict[str, Dict[FrozenSet[str], Union[KeyValues, IODef]]]
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

    fgd = load_database(dbase, extra_db)

    if engine_mode:
        # In engine mode, we don't care about specific games.
        print('Collapsing bases...')
        fgd.collapse_bases()

        # Cache these constant sets.
        tags_empty = frozenset('')
        tags_engine = frozenset({'ENGINE'})

        print('Merging tags...')
        for ent in fgd:
            # Strip applies-to helper and ordering helper.
            ent.helpers[:] = [
                helper for helper in ent.helpers
                if not helper.IS_EXTENSION
            ]
            value: Union[IODef, KeyValues]
            category: Dict[str, Dict[FrozenSet[str], Union[IODef, KeyValues]]]
            for category in [ent.inputs, ent.outputs, ent.keyvalues]:
                # For each category, check for what value we want to keep.
                # If only one, we keep that.
                # If there's an "ENGINE" tag, that's specifically for us.
                # Otherwise, warn if there's a type conflict.
                # If the final value is choices, warn too (not really a type).
                for key, tag_map in category.items():
                    if len(tag_map) == 1:
                        [value] = tag_map.values()
                    elif tags_engine in tag_map:
                        value = tag_map[tags_engine]
                        if value.type is ValueTypes.CHOICES:
                            raise ValueError(
                                '{}.{}: Engine tags cannot be '
                                'CHOICES!'.format(ent.classname, key)
                            )
                    else:
                        # More than one tag.
                        # IODef and KeyValues have a type attr.
                        types = {val.type for val in tag_map.values()}
                        if len(types) > 2:
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
                                for choice_val, name, tag in value.val_list:
                                    int(choice_val)
                            except ValueError:
                                # Not all are ints, it's a string.
                                value.type = ValueTypes.STRING
                            else:
                                value.type = ValueTypes.INT
                            value.val_list = None

                    # Blank this, it's not that useful.
                    value.desc = ''

                    category[key] = {tags_empty: value}

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

    for poly_tag, polyfill in POLYFILLS:
        if not poly_tag or poly_tag in tags:
            polyfill(fgd)

    print('Applying helpers to child entities and optimising...')
    for ent in fgd.entities.values():
        # Merge them together.
        helpers = []
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
    used_bases = set()  # type: Set[EntityDef]
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
        ent.bases = [
            base
            for base in ent.bases
            if base.type is not EntityTypes.BASE or base in used_bases
        ]

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

    print('Exporting...')

    if as_binary:
        with open(output_path, 'wb') as bin_f, LZMAFile(bin_f, 'w') as comp:
            fgd.serialise(comp)
    else:
        with open(output_path, 'w', encoding='iso-8859-1') as txt_f:
            fgd.export(txt_f)
            # BEE2 compatibility, don't make it run.
            if 'P2' in tags:
                txt_f.write('\n// BEE 2 EDIT FLAG = 0 \n')


def action_visgroup(
    dbase: Path,
    extra_loc: Path,
    dest: Path,
) -> None:
    """Dump all auto-visgroups into the specified file, using a custom format."""
    fgd = load_database(dbase, extra_loc, fgd_vis=True)

    # TODO: This shouldn't be copied from fgd.export(), need to make the
    #  parenting invariant guaranteed by the classes.
    vis_by_parent = defaultdict(set)  # type: Dict[str, Set[AutoVisgroup]]

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
        description="Manage a set of unified FGDs, sharing configs "
                    "between engine versions.",

    )
    parser.add_argument(
        "-d", "--database",
        default="fgd/",
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

    if result.extra_db is not None:
        extra_db = Path(result.extra_db).resolve()  # type: Optional[Path]
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
                parser.error(
                    'Invalid tag "{}"! Allowed tags: \n'.format(tag) +
                    format_all_tags()
                )
        action_export(
            dbase,
            extra_db,
            tags,
            result.output,
            result.binary,
            result.engine,
        )
    elif result.mode in ("c", "count"):
        action_count(dbase, extra_db)
    elif result.mode in ("visgroup", "v", "vis"):
        action_visgroup(dbase, extra_db, result.output)
    else:
        raise AssertionError("Unknown mode! (" + result.mode + ")")


if __name__ == '__main__':
    main(sys.argv[1:])

    #for game in GAME_ORDER:
    #    print('\n'+ game + ':')
    #    main(['export', '-o', 'fgd_out/' + game + '.fgd', game])
