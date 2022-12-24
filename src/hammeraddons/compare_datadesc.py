"""Compare the FGD database to datadesc dumps."""
from typing import Dict, FrozenSet, Literal, Optional, Set
from pathlib import Path
import re

from srctools.fgd import EntityDef

from unify_fgd import expand_tags, load_database, match_tags

repo_root = Path(__file__).parents[2]


# Match: "- m_iClassname (Offset 92) (Save|Key)(4 Bytes) - classname"
regex = re.compile(
    r' *- ([^)]+) *\(Offset [0-9]+\) *\(([a-zA-Z|]*)\)\([0-9]+ Bytes\) *(?:- *([a-zA-Z0-9_ ]+))?'
)


def check_tagsmap(tagsmap: Dict[FrozenSet[str], object], valid_tags: FrozenSet[str]) -> bool:
    """Check if the tagsmap contains something that matches this tag."""
    for tag in tagsmap:
        if match_tags(valid_tags, tag):
            return False
    return True


def get_val(
    ent: EntityDef,
    kv_or_io: Literal['keyvalues', 'inputs', 'outputs'],
    name: str,
) -> Dict[FrozenSet[str], object]:
    """Check this ent and parents for this key/i/o."""
    name = name.casefold()
    try:
        return getattr(ent, kv_or_io)[name]
    except KeyError:
        pass
    for base in ent.iter_bases():
        try:
            return getattr(base, kv_or_io)[name]
        except KeyError:
            pass
    # Last ditch: if on BaseEntity, ignore for now.
    return getattr(base_ent, kv_or_io)[name]


def check_datadesc(filename: str, tags: FrozenSet[str]) -> None:
    """Check a specific datadesc."""
    print(f'\nChecking {filename}datamap.txt ... ')
    tags = expand_tags(tags) | {'ENGINE', 'COMPLETE'}
    print('Expanded tags:', sorted(tags))

    bad_ents: Set[str] = set()
    message_count = 0
    classname = '?????'

    def msg(*args: object) -> None:
        """Write a message to the report file and stdout."""
        nonlocal message_count
        message_count += 1
        print(*args, file=msgfile)
        bad_ents.add(classname)

    with open(Path(repo_root, 'db', 'datamaps', filename + 'datamap.txt')) as f, open(Path(repo_root, 'db', 'reports', filename + '.txt'), 'w') as msgfile:
        cur_ent: Optional[EntityDef]  # Deliberately uninitialised.
        for line in f:
            if line.startswith('//') or not line.strip():
                continue
            line = line.rstrip()

            if not line.startswith((' ', '-')):  # Entity classname.
                real_name, classname = map(str.strip, line.split('-', 2))
                try:
                    cur_ent = database[classname]
                except KeyError:
                    msg(f'"{classname}" has no FGD!')
                    bad_ents.add(classname)
                    cur_ent = None
            elif line.strip().startswith('-'):
                # A value.
                match = regex.match(line)
                if match is not None:
                    real_name, flags, name = match.groups()
                    if name is None:  # Internal only.
                        continue
                    name = name.replace(' ', '')
                    try:
                        if cur_ent is None:
                            # msg(f'[{classname}] Missing member:', name)
                            continue
                    except UnboundLocalError:
                        raise ValueError('KV coming before first ent??')
                    if 'Output' in flags:
                        try:
                            tagsmap = get_val(cur_ent, 'outputs', name)
                        except KeyError:
                            msg(f'[{classname}] Missing output "{name}" -> ', real_name, flags)
                            continue
                        if check_tagsmap(tagsmap, tags):
                            msg(f'[{classname}] Tag mismatch for output "{name}" - allowed = {list(tagsmap)}')
                    elif 'Key' in flags:  # Key|Output means key is ignored.
                        # present in all ents, has mismatches in FGD etc.
                        if name.casefold() in ['target', 'teamnum', 'view', 'max', 'teamnum', 'solid', 'spawnflags', 'model']:
                            continue
                        try:
                            tagsmap = get_val(cur_ent, 'keyvalues', name)
                        except KeyError:
                            msg(f'[{classname}] Missing KV "{name}" -> ', real_name, flags)
                            continue
                        if check_tagsmap(tagsmap, tags):
                            msg(f'[{classname}] Tag mismatch for key "{name}" - allowed = {list(tagsmap)}')

                    if 'Input' in flags:
                        try:
                            tagsmap = get_val(cur_ent, 'inputs', name)
                        except KeyError:
                            msg(f'[{classname}] Missing input "{name}" -> ', real_name, flags)
                            continue
                        if check_tagsmap(tagsmap, tags):
                            msg(f'[{classname}] Tag mismatch for input "{name}" - allowed = {list(tagsmap)}')
                elif 'null' in line:
                    continue  # Anomalous entry.
                else:
                    msg('No match?', repr(line))
            elif 'Sub-Class Table' in line:
                continue  # We don't care about subclass distinctions.
            else:
                msg('Unrecognised line:', line)
        msgfile.write(f'\n\nTotal: {len(bad_ents)} ents')
        print(f'Total: {len(bad_ents)} ents, errors: {message_count}')


database, base_ent = load_database(repo_root / 'fgd/')

check_datadesc('hl2', frozenset({'HL2'}))
check_datadesc('episodic', frozenset({'EP1', 'EP2'}))
check_datadesc('portal', frozenset({'P1'}))
check_datadesc('portal2', frozenset({'P2'}))
check_datadesc('tf2', frozenset({'TF2'}))
check_datadesc('csgo', frozenset({'CSGO'}))
check_datadesc('l4d', frozenset({'L4D'}))
check_datadesc('l4d2', frozenset({'L4D2'}))
