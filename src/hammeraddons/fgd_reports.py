"""Defines various reports which analyse the database, identifying possible improvements."""
from collections import Counter, defaultdict
from typing import TextIO

from pathlib import Path
from pprint import pprint, pformat

from collections.abc import MutableMapping

from srctools import FGD
from srctools.fgd import (
    EntityTypes, ValueTypes, EntityDef, ResourceCtx,
    TagsSet, match_tags, HelperModel, HelperSprite,
)

from .unify_fgd import (
    ALL_GAMES, ALL_MODS, SNIPPET_USED, GAME_ORDER, expand_tags, get_appliesto,
    UNIQUE_HELPERS,
)


def report_counts(fgd: FGD, report_dir: Path) -> MutableMapping[str, set[str]]:
    """Count how many of each entity type exist.

    This also returns a mapping from games to the entities contained within.
    """
    count_base: dict[str, int] = Counter()
    count_point: dict[str, int] = Counter()
    count_brush: dict[str, int] = Counter()

    all_tags = {
        tag.lstrip('+-!').upper()
        for ent in fgd
        for tag in get_appliesto(ent)
    }

    games = (ALL_GAMES | ALL_MODS) & all_tags

    print(f'Defined games: {pformat(games, compact=True)}')

    expanded: dict[str, TagsSet] = {
        # Opt into complete list, since we're checking against engine dumps.
        game: expand_tags(frozenset({game, 'COMPLETE'}))
        for game in ALL_GAMES | ALL_MODS
    }
    expanded['ALL'] = frozenset()

    game_classes: MutableMapping[tuple[str, str], set[str]] = defaultdict(set)
    base_uses: MutableMapping[str, set[str]] = defaultdict(set)
    all_ents: MutableMapping[str, set[str]] = defaultdict(set)

    kv_counts: dict[tuple, list[tuple]] = defaultdict(list)
    inp_counts: dict[tuple, list[tuple]] = defaultdict(list)
    out_counts: dict[tuple, list[tuple]] = defaultdict(list)
    desc_counts: dict[tuple, list[tuple]] = defaultdict(list)
    val_list_counts: dict[tuple, list[tuple]] = defaultdict(list)

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
            assert isinstance(base, EntityDef), (ent, ent.bases)
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

        if ent.classname in SNIPPET_USED:
            # This entity does use snippets already, don't count it.
            continue

        for name, kv_map in ent.keyvalues.items():
            for tags, kv in kv_map.items():
                if 'ENGINE' in tags or '+ENGINE' in tags or kv.type is ValueTypes.SPAWNFLAGS:
                    continue
                if kv.desc:  # Blank is not a duplicate!
                    desc_counts[kv.desc, ].append((ent.classname, name))
                kv_counts[
                    kv.name, kv.type, (tuple(kv.val_list) if kv.val_list is not None else ()), kv.desc, kv.default,
                ].append((ent.classname, name, kv.desc))
                if kv.val_list is not None:
                    val_list_counts[tuple(kv.val_list)].append((ent.classname, name))
        for name, io_map in ent.inputs.items():
            for tags, io in io_map.items():
                if 'ENGINE' in tags or '+ENGINE' in tags:
                    continue
                inp_counts[io.name, io.type, io.desc].append((ent.classname, name, io.desc))
        for name, io_map in ent.outputs.items():
            for tags, io in io_map.items():
                if 'ENGINE' in tags or '+ENGINE' in tags:
                    continue
                out_counts[io.name, io.type, io.desc].append((ent.classname, name, io.desc))

    all_games: set[str] = {*count_base, *count_point, *count_brush}

    def ordering(game: str) -> tuple:
        """Put ALL at the start, mods at the end."""
        if game == 'ALL':
            return (0, 0)
        try:
            return (1, GAME_ORDER.index(game))
        except ValueError:
            return (2, game)  # Mods

    game_order = sorted(all_games, key=ordering)

    row_temp = '{:^9} | {:^6} | {:^6} | {:^6}\n'
    header = row_temp.format('Game', 'Base', 'Point', 'Brush')
    print('Counted entities.')

    with open(report_dir / 'counts.txt', 'w', encoding='utf8') as f:
        f.write(header)
        print('-' * len(header), file=f)

        for game in game_order:
            f.write(row_temp.format(
                game,
                count_base[game],
                count_point[game],
                count_brush[game],
            ))

        f.write('\n\nBases:\n')
        for base, used in sorted(base_uses.items(), key=lambda x: (len(x[1]), x[0])):
            ent = fgd[base]
            if ent.type is EntityTypes.BASE and (
                ent.keyvalues or ent.outputs or ent.inputs
            ):
                f.write(f'{base} {len(used)} {used if len(used) == 1 else '...'}\n')

    for kind_name, count_map in (
        ('keyvalues', kv_counts),
        ('inputs', inp_counts),
        ('outputs', out_counts),
        ('val list', val_list_counts),
        ('desc', desc_counts)
    ):
        count = 0
        with open(report_dir / f'duplicate_{kind_name}.txt', 'w', encoding='utf8') as f:
            for key, info in sorted(count_map.items(), key=lambda v: len(v[1]), reverse=True):
                if len(info) <= 2:
                    continue
                f.write(f'{len(info):02}: {key[:64]!r} -> {info}\n')
                count += 1
        print(f'{count} duplicate {kind_name}.')

    return all_ents


def report_factories(
    all_ents: MutableMapping[str, set[str]],
    *,
    report_dir: Path, factories_folder: Path,
) -> None:
    """Use a dump of entity factories from games to check for missing/extra ents."""
    all_classes = set()
    used_classes = set()
    for dump_path in factories_folder.glob('*.txt'):
        dump_classes = set()
        with dump_path.open() as f:
            for line in f:
                line = line.casefold().strip()
                if line.isspace():
                    continue
                # Strata's output has lines like 'hl2:weapon_crowbar'. We don't care right now.
                if ':' in line:
                    line = line.split(':', 1)[1]
                dump_classes.add(line)
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
        all_classes |= defined_classes
        used_classes |= dump_classes
        with open(report_dir / f'factories_{game.lower()}.txt', 'w', encoding='utf8') as rep_f:
            rep_f.write('Extraneous definitions: \n')
            pprint(sorted(extra), rep_f, compact=True)
            rep_f.write('\n\nMissing definitions: \n')
            pprint(sorted(missing), rep_f, compact=True)

    unused = all_classes - used_classes
    with open(report_dir / 'factories_unused.txt', 'w', encoding='utf8') as rep_f:
        pprint(sorted(unused), rep_f, compact=True)
    print(f'Checked entity factories. {len(unused)} totally unused.')


def report_undefined_resources(fgd: FGD, report_dir: Path) -> None:
    """Identify entities without class resources defined."""
    missing_count = defined_count = empty_count = 0
    not_in_engine = {'-ENGINE', '!ENGINE', 'SRCTOOLS', '+SRCTOOLS'}
    class_res = defaultdict(list)
    for clsname in sorted(fgd.entities):
        ent = fgd.entities[clsname]
        if ent.type is EntityTypes.BASE or ent.is_alias:
            continue
        appliesto = get_appliesto(ent)

        if not not_in_engine.isdisjoint(appliesto):
            continue
        if ent.resources_defined():
            defined_count += 1
            if len(ent.resources) == 0:
                empty_count += 1
        else:
            class_res[frozenset(appliesto)].append(ent.classname)
            missing_count += 1

    with open(report_dir / 'undefined_resources.txt', 'w', encoding='utf8') as f:
        for tags_list, classnames in class_res.items():
            classnames.sort()
            f.write(f'{', '.join(tags_list)} = {pformat(classnames, compact=True)}\n')
        summary = (
            f'\nMissing: {missing_count}, '
            f'Defined: {defined_count} = {defined_count/(missing_count + defined_count):.2%}, empty={empty_count}'
        )
        print(summary)
        f.write(summary + '\n')


def report_missing_resources(fgd: FGD, report_dir: Path) -> None:
    """Report resource references which don't resolve."""
    count = 0

    def report(msg: str) -> None:
        """Report errors in resources."""
        nonlocal count
        count += 1
        f.write(f'Ent {ent.classname} res error: {msg}\n')
    res_ctx = ResourceCtx(fgd=fgd)

    with open(report_dir / 'missing_resources.txt', 'w', encoding='utf8') as f:
        for ent in fgd.entities.values():
            # Get them all, checking validity in the process.
            for _ in ent.get_resources(res_ctx, ent=None, on_error=report):
                pass

    print(f'Found {count} missing resources.')


def check_ent_sprites(used: dict[str, list[str]], f: TextIO, ent: EntityDef) -> None:
    """Check if the specified entity has a unique sprite."""
    mdl: str | None = None
    sprite: str | None = None
    for helper in ent.helpers:
        if type(helper) in UNIQUE_HELPERS:
            return  # Specialised helper is sufficient.
        if isinstance(helper, HelperModel):
            if helper.model is None and 'model' in ent.kv:
                return  # Model is customisable.
            mdl = helper.model
        if isinstance(helper, HelperSprite):
            if helper.mat is None:
                f.write(f'{ent.classname}: {helper!r}???\n')
            sprite = helper.mat
    # If both model and sprite, allow model to be duplicate.
    if mdl and sprite:
        display = sprite
    elif mdl:
        display = mdl
    elif sprite:
        display = sprite
    else:
        tags = get_appliesto(ent)
        if 'ENGINE' not in tags and '+ENGINE' not in tags:
            f.write(f'{ent.classname}: No sprite/model? {pformat(ent.helpers)}\n')
        return
    used[display].append(ent.classname)


def report_helper_reuse(fgd: FGD, report_dir: Path) -> None:
    """Report missing or reused helpers."""
    mdl_or_sprites: dict[str, list[str]] = defaultdict(list)
    with open(report_dir / 'helper_reuse.txt', 'w', encoding='utf8') as f:
        for ent in fgd:
            if ent.type is not EntityTypes.BASE and ent.type is not EntityTypes.BRUSH and not ent.is_alias:
                check_ent_sprites(mdl_or_sprites, f, ent)
        for resource, classes in mdl_or_sprites.items():
            if len(classes) > 1:
                classes.sort()
                f.write(f'Reused {resource}: {classes}\n')
    print('Checked helper reuse.')


def report_repeated_bases(fgd: FGD, report_dir: Path) -> None:
    """Report entities which include the same base multiple times in their hierachy."""

    def check_parents(done: set[EntityDef], repeat: set[EntityDef], ent: EntityDef) -> None:
        """Recursively check a hierachy."""
        if ent in done:
            repeat.add(ent)
        else:
            done.add(ent)
            for base in ent.bases:
                assert isinstance(base, EntityDef), (ent, ent.bases)
                check_parents(done, repeat, base)

    with open(report_dir / 'repeated_bases.txt', 'w', encoding='utf8') as f:
        for ent in fgd:
            done: set[EntityDef] = set()
            repeat: set[EntityDef] = set()
            check_parents(done, repeat, ent)
            if repeat:
                f.write(
                    f'Repeated bases: {ent.classname} = '
                    f'{[ent.classname for ent in repeat]}, '
                    f'all={[ent.classname for ent in done]}'
                )
    print('Checked repeated bases.')
