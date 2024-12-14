"""Apply transformations that work on (almost) all entities."""
from typing import Dict, List, Tuple
from collections import defaultdict
import itertools

from srctools import Output
from srctools.logger import get_logger
from srctools.packlist import FileType

from hammeraddons.bsp_transform import Context, trans


LOGGER = get_logger(__name__)


@trans('Attachment Points')
def att_points(ctx: Context) -> None:
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
def vscript_init_code(ctx: Context) -> None:
    """Add vscript_init_code keyvalues.

    The specified code is appended as a script file to the end of the scripts.
    vscript_init_code2, 3 etc will also be added in order if they exist.
    """
    for ent in ctx.vmf.entities:
        code = ent.pop('vscript_init_code', '')

        if not code:
            continue

        for i in itertools.count(2):
            extra = ent.pop('vscript_init_code' + str(i), '')
            if not extra:
                break
            code += '\n' + extra

        ctx.add_code(ent, code)


@trans('VScript RunScript Inputs', priority=500)
def vscript_runscript_inputs(ctx: Context) -> None:
    """Handle RunScript* inputs.

    For RunScriptCode, allow using quotes in the parameter.  TF2 implements this in game code,
    so we don't need to do it there.

    This is done by using ` as a replacement for double-quotes,
    then synthesising a script file and using RunScriptFile to execute it.
    For RunScriptFile, ensure the file is packed.
    """
    in_tf2 = 'TF2' in ctx.tags
    for ent in ctx.vmf.entities:
        for out in ent.outputs:
            inp_name = out.input.casefold()
            if inp_name == 'runscriptfile':
                ctx.pack.pack_file('scripts/vscripts/' + out.params, FileType.VSCRIPT_SQUIRREL)
            elif inp_name == 'runscriptcode' and not in_tf2 and '`' in out.params:
                out.params = ctx.pack.inject_vscript(out.params.replace('`', '"'))
                out.input = 'RunScriptFile'


@trans('Optimise logic_auto', priority=50)
def optimise_logic_auto(ctx: Context) -> None:
    """Merge logic_auto entities to simplify the map."""

    # (global state) -> outputs
    states: Dict[Tuple[str, bool], List[Output]] = defaultdict(list)

    for auto in ctx.vmf.by_class['logic_auto']:
        # If the auto uses any keys that we don't recognise, leave it alone.
        # These catch stuff like it being named and in a template,
        # VScript, or any other hijinks.
        if any(
            value and key.casefold() not in {
                'origin', 'angles', 'spawnflags',
                'globalstate',
            }
            for key, value in auto.items()
        ):
            continue
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
            spawnflags=only_once,
        ).outputs = outputs


@trans('Strip Entities', priority=50)
def strip_ents(ctx: Context) -> None:
    """Strip useless entities from the map."""
    for clsname in [
        # None of these are defined by the engine itself.
        # If present they're useless.
        'hammer_notes',
        'func_instance_parms',
        'func_instance_origin',
        'comp_player_input_helper',
        'hammer_model',
    ]:
        for ent in ctx.vmf.by_class[clsname]:
            ent.remove()

    # Strip the divider keyvalues in the FGDs.
    to_remove: List[str] = []
    for ent in ctx.vmf.entities:
        to_remove.clear()
        for key, value in ent.items():
            if 'divider' in key and value == "":
                to_remove.append(key)
        for key in to_remove:
            del ent[key]
