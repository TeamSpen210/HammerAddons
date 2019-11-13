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

    ('TF2',  'Team Fortress 2'),
    ('P1', 'Portal'),
    ('L4D', 'Left 4 Dead'),
    ('L4D2', 'Left 4 Dead 2'),
    ('ASW', 'Alien Swam'),
    ('P2', 'Portal 2'),
    ('CSGO', 'Counter-Strike Global Offensive'),

    ('SFM', 'Source Filmmaker'),
    ('DOTA2', 'Dota 2'),
    ('PUNT', 'PUNT'),
    ('P2DES', 'Portal 2: Desolation'),
]  # type: List[Tuple[str, str]]

GAME_ORDER = [game for game, desc in GAMES]
GAME_NAME = dict(GAMES)

# Specific features that are backported to various games.

FEATURES = {
    # 2013 engine backports this.
    'HL2': {'INSTANCING'},
    'EP1': {'INSTANCING'},
    'EP2': {'INSTANCING'},

    'MBASE': {'HL2', 'EP1', 'EP2', 'INSTANCING'},
    
    'L4D': {'INSTANCING'},
    'L4D2': {'INSTANCING', 'INST_IO', 'VSCRIPT'},
    'TF2': {'INSTANCING', 'PROP_SCALING'},
    'ASW': {'INSTANCING', 'INST_IO', 'VSCRIPT'},
    'P2': {'INSTANCING', 'INST_IO', 'VSCRIPT'},
    'CSGO': {'INSTANCING', 'INST_IO', 'PROP_SCALING', 'VSCRIPT', 'PROPCOMBINE'},
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
  'BEE2', # BEEmod's templates.
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


def _polyfill(tag: str) -> Callable[[PolyfillFuncT], PolyfillFuncT]:
    """Register a polyfill, which backports newer FGD syntax to older engines."""
    def deco(func: PolyfillFuncT) -> PolyfillFuncT:
        POLYFILLS.append((tag.upper(), func))
        return func
    return deco


@_polyfill('until_p1')
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


@_polyfill('until_p1')
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


def load_database(dbase: Path, extra_loc: Path=None) -> FGD:
    """Load the entire database from disk."""
    print('Loading database:')
    fgd = FGD()

    fgd.map_size_min = -16384
    fgd.map_size_max = 16384

    with RawFileSystem(str(dbase)) as fsys:
        for file in dbase.rglob("*.fgd"):
            fgd.parse_file(
                fsys,
                fsys[str(file.relative_to(dbase))],
                eval_bases=False,
            )
            print('.', end='', flush=True)

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
    return fgd


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
    arg_list = sorted(applies_to)
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


def action_count(dbase: Path, extra_db: Optional[Path]) -> None:
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

    for ent in fgd:
        if ent.type is EntityTypes.BASE:
            counter = count_base
            typ = 'Base'
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
            if match_tags(appliesto, tags):
                counter[game] += 1
                game_classes[game, typ].add(ent.classname)
                has_ent.add(game)

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

    game_order = ['ALL'] + sorted(all_games - {'ALL'})

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

    print('\n\nBases:')
    for base, count in sorted(base_uses.items(), key=lambda x: (len(x[1]), x[0])):
        if fgd[base].type is EntityTypes.BASE:
            print(base, len(count), count if len(count) == 1 else '...')


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
        if not applies_to:
            ent.helpers.remove((HelperTypes.EXT_APPLIES_TO, []))

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

    print('Applying helpers to child entities...')
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

    # Helpers aren't inherited, so this isn't useful anymore.
    for ent in fgd.entities.values():
        if ent.type is EntityTypes.BASE:
            ent.helpers.clear()

    print('Exporting...')

    if as_binary:
        with open(output_path, 'wb') as bin_f, LZMAFile(bin_f, 'w') as comp:
            fgd.serialise(comp)
    else:
        with open(output_path, 'w') as txt_f:
            fgd.export(txt_f)
            # BEE2 compatibility, don't make it run.
            txt_f.write('\n// BEE 2 EDIT FLAG = 0 \n')


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
    else:
        raise AssertionError("Unknown mode! (" + result.mode + ")")


if __name__ == '__main__':
    main(sys.argv[1:])

    #for game in GAME_ORDER:
    #    print('\n'+ game + ':')
    #    main(['export', '-o', 'fgd_out/' + game + '.fgd', game])
