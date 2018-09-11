"""Implements "unified" FGD files.

This allows sharing definitions among different engine versions.
"""
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Set, FrozenSet, Union, Dict

from srctools.fgd import (
    FGD, validate_tags, match_tags,
    EntityDef, EntityTypes,
    HelperTypes, IODef,
    KeyValues,
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
    'L4D': 'INSTANCING'.split(), 
    'TF2': 'INSTANCING PROP_SCALING'.split(),
    'ASW': 'INSTANCING VSCRIPT'.split(),
    'P2': 'INSTANCING VSCRIPT'.split(),
    'CSGO': 'INSTANCING PROP_SCALING VSCRIPT'.split(),
    'P2DES': 'INSTANCING PROP_SCALING VSCRIPT'.split(),
}

ALL_TAGS = set()  # type: Set[str]
ALL_TAGS.update(GAME_ORDER)
ALL_TAGS.update(tag.upper() for t in FEATURES.values() for tag in t)
ALL_TAGS.update('SINCE_' + t.upper() for t in GAME_ORDER)
ALL_TAGS.update('UNTIL_' + t.upper() for t in GAME_ORDER)

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
            pos = GAME_ORDER.index(tag)
        except IndexError:
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
    elif ent.type is EntityTypes.BRUSH:
        folder = 'brush'
    else:
        folder = 'point/'

    # if '_' in ent.classname:
    #     folder += '/' + ent.classname.split('_', 1)[0]

    return '{}/{}.fgd'.format(folder, ent.classname)


def load_database(dbase: Path) -> FGD:
    """Load the entire database from disk."""
    print('Loading database...')
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
            print('.', end='')
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
    applies_to = set()
    for i, (helper_type, args) in enumerate(ent.helpers):
        if helper_type is HelperTypes.EXT_APPLIES_TO:
            if pos is None:
                pos = i
            applies_to.update(args)

    if pos is None:
        pos = 0
    arg_list = sorted(applies_to)
    ent.helpers[:] = [
        tup for tup in
        ent.helpers
        if tup[0] is not HelperTypes.EXT_APPLIES_TO
    ]
    ent.helpers.insert(pos, (HelperTypes.EXT_APPLIES_TO, arg_list))
    return arg_list


def add_tag(tags: FrozenSet[str], new_tag: str) -> FrozenSet[str]:
    """Modify these tags such that they allow the new tag."""
    tag_set = set(tags)
    if new_tag.startswith(('!', '-')):
        tag_set.discard(new_tag[1:])
        tag_set.add(new_tag)
    else:
        tag_set.discard('!' + new_tag.upper())
        tag_set.discard('-' + new_tag.upper())
        if ('+' + new_tag.upper()) not in tag_set:
            tag_set.add(new_tag.upper())

    return frozenset(tag_set)


def action_import(
    dbase: Path,
    engine_tag: str,
    fgd_paths: List[Path],
) -> None:
    """Import an FGD file, adding differences to the unified files."""
    new_fgd = FGD()
    print('Using tag "{}"'.format(engine_tag))

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
            to_export = old_fgd
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
            # We just write this entity in.
            to_export = new_ent

        applies_to = get_appliesto(ent)
        if engine_tag not in applies_to:
            applies_to.append(engine_tag)

        with open(path, 'w') as f:
            to_export.export(f)

        print('.', end='', flush=True)
    print()


def action_export(
    dbase: Path,
    tags: FrozenSet[str],
    output_path: Path,
) -> None:
    """Create an FGD file using the given tags."""
    tags = expand_tags(tags)

    print('Tags expanded to: {}'.format(', '.join(tags)))

    fgd = load_database(dbase)

    print('Culling incompatible entities...')

    ents = list(fgd.entities.values())
    fgd.entities.clear()

    for ent in ents:
        applies_to = get_appliesto(ent)
        if match_tags(tags, applies_to):
            fgd.entities[ent.classname] = ent

            # Strip applies-to helper.
            ent.helpers[:] = [
                helper for helper in ent.helpers
                if helper[0] is not HelperTypes.EXT_APPLIES_TO
            ]
            ent.strip_tags(tags)

    print('Culled entities, merging bases...')

    fgd.collapse_bases()

    print('Exporting...')

    fgd.entities = {
        clsname: ent
        for clsname, ent in fgd.entities.items()
        if ent.type is not EntityTypes.BASE
    }

    with open(output_path, 'w') as f:
        fgd.export(f)


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
    subparsers = parser.add_subparsers(dest="mode")

    parser_exp = subparsers.add_parser(
        "export",
        help=action_export.__doc__,
        aliases=["exp", "e"],
    )

    parser_exp.add_argument(
        "-o", "--output",
        default="output.fgd",
        help="Destination FGD filename."
    )
    parser_exp.add_argument(
        "tags",
        choices=ALL_TAGS,
        nargs="+",
        help="Tags to include in the output.",
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

    if result.mode in ("import", "imp", "i"):
        action_import(
            dbase,
            result.engine,
            result.fgd,
        )
    elif result.mode in ("export", "exp", "e"):
        action_export(
            dbase,
            validate_tags(result.tags),
            result.output,
        )
    else:
        raise AssertionError("Unknown mode! (" + result.mode + ")")


if __name__ == '__main__':
    # main(sys.argv[1:])

    for game in GAME_ORDER:
        print('\n'+ game + ':')
        main(['export', '-o', 'fgd_out/' + game + '.fgd', game])
