"""Apply transformations that work on (almost) all entities."""

"""Implements various brush entities."""
from srctools.bsp_transform import trans, Context
from srctools.logger import get_logger

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
    """
    for ent in ctx.vmf.entities:
        code = ent['vscript_init_code']
        if not code:
            continue
        init_scripts = ent['vscripts'].split()
        init_scripts.append(ctx.pack.inject_file(
            code.replace('`', '"').encode('utf8'),
            'scripts/vscripts/inject',
            'nut'
        )[17:])  # Don't include scripts/vscripts/
        ent['vscripts'] = ' '.join(init_scripts)
        del ent['vscript_init_code']


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
