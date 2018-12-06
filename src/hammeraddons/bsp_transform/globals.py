"""Apply transformations that work on (almost) all entities."""
import itertools
from collections import defaultdict
from typing import Dict, Tuple, List

from srctools.bsp_transform import trans, Context
from srctools.logger import get_logger
from srctools import Output


LOGGER = get_logger(__name__)


@trans('Attachment Points')
def att_points(ctx: Context):
    """Allow setting attachment points in a separate field to the parent name."""
    for ent in ctx.vmf.entities:
        if not ent['parent_attachment_point']:
            continue
        parent = ent['parentname'].rsplit(',', 1)[0]

        if not parent:
            LOGGER.warning(
                'No parent, but attachment point set for "{}"? ({})',
                ent['targetname'],
                ent['origin'],
            )
            continue

        ent['parentname'] = parent + ',' + ent['parent_attachment_point']


@trans('VScript Init Code')
def vscript_init_code(ctx: Context):
    """Add vscript_init_code keyvalues.

    The specified code is appended as a script file to the end of the scripts.
    vscript_init_code2, 3 etc will also be added in order if they exist.
    """
    for ent in ctx.vmf.entities:
        code = ent.keys.pop('vscript_init_code', '')

        if not code:
            continue

        for i in itertools.count(2):
            extra = ent.keys.pop('vscript_init_code' + str(i), '')
            if not extra:
                break
            code += '\n' + extra

        init_scripts = ent['vscripts'].split()
        init_scripts.append(ctx.pack.inject_file(
            code.replace('`', '"').encode('utf8'),
            'scripts/vscripts/inject',
            'nut'
        )[17:])  # Don't include scripts/vscripts/
        ent['vscripts'] = ' '.join(init_scripts)


@trans('VScript RunScriptCode Strings')
def vscript_runscriptcode_strings(ctx: Context):
    """Allow writing strings in RunScriptCode inputs.

    This is done by using ` as a replacement for double-quotes,
    then synthesising a script file and using RunScriptFile to execute it.
    """
    for ent in ctx.vmf.entities:
        for out in ent.outputs:
            if out.input.casefold() != 'runscriptcode':
                continue
            if '`' not in out.params:
                continue
            out.params = ctx.pack.inject_file(
                out.params.replace('`', '"').encode('utf8'),
                'scripts/vscripts/inject',
                'nut'
            )[17:]  # Don't include scripts/vscripts/
            out.input = 'RunScriptFile'


@trans('Optimise logic_auto')
def optimise_logic_auto(ctx: Context):
    """Merge logic_auto entities to simplify the map."""

    # (global state) -> outputs
    states = defaultdict(list)  # type: Dict[Tuple[str, bool], List[Output]]

    for auto in ctx.vmf.by_class['logic_auto']:
        auto.remove()
        state = auto['globalstate', '']
        only_once = auto['spawnflags', '0'] == '1'
        for out in auto.outputs:
            # We know OnMapSpawn only happens once.
            if out.output.casefold() == 'onmapspawn' or only_once:
                out.only_once = True
            states[state, out.only_once].append(out)

    for (state, only_once), outputs in states.items():
        ctx.vmf.create_ent(
            classname='logic_auto',
            globalstate=state,
            origin='0 0 0',
            spawnflags=int(only_once),
        ).outputs = outputs
