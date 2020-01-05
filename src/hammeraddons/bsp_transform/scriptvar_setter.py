import re
from typing import Dict, Tuple,  Optional, Callable

from srctools.bsp_transform import trans, Context
from srctools.logger import get_logger
from srctools import Entity, Vec, conv_float


LOGGER = get_logger(__name__)


def vs_vec(vec: Vec) -> str:
    """Convert the provided Vec into a VScript Vector constructor code."""
    return 'Vector({})'.format(vec.join(', '))


@trans('VScript Scriptvar')
def comp_scriptvar(ctx: Context):
    """An entity to allow setting VScript variables to information from the map."""
    # {ent: {variable: {index: value}}}
    set_vars = {}  # type: Dict[Entity, Dict[str, Dict[Optional[int], str]]]

    for comp_ent in ctx.vmf.by_class['comp_scriptvar']:
        comp_ent.remove()
        var_name = orig_var_name = comp_ent['variable']
        index = None

        parsed_match = re.fullmatch(r'\s*([^[]+)\[([0-9]+)\]\s*', var_name)
        if parsed_match:
            var_name, index_str = parsed_match.groups()
            try:
                index = int(index_str)
            except (TypeError, ValueError):
                LOGGER.warning(
                    'Invalid variable index in '
                    'comp_scriptvar at {} targetting "{}"!',
                    comp_ent['origin'], comp_ent['target']
                )
                continue
        elif '[' in var_name or ']' in var_name:
            LOGGER.warning(
                'Unparsable variable[index] in '
                'comp_scriptvar at {} targetting "{}"!',
                comp_ent['origin'], comp_ent['target']
            )
            continue

        ref_name = comp_ent['ref']
        ref_ent = comp_ent
        if ref_name:
            for ref_ent in ctx.vmf.search(ref_name):
                break
            else:
                LOGGER.warning(
                    'Can\'t find ref entity named "{}" '
                    'for comp_scriptvar at <{}>!',
                    ref_name,
                    comp_ent['origin'],
                )

        try:
            mode_func = globals()['mode_' + comp_ent['mode']]
        except KeyError:
            LOGGER.warning(
                'Invalid mode "{}" in '
                'comp_scriptvar at {} targetting "{}"!',
                comp_ent['mode'], comp_ent['origin'], comp_ent['target'],
            )
            continue
        else:
            code = mode_func(comp_ent, ref_ent)

        ent = None
        for ent in ctx.vmf.search(comp_ent['target']):
            ind_dict = set_vars.setdefault(ent, {}).setdefault(var_name, {})
            if index in ind_dict:
                LOGGER.warning(
                    'comp_scriptvar at {} overwrote '
                    'the variable {}  on "{}"!',
                    comp_ent['origin'], orig_var_name, comp_ent['target'],
                )
            ind_dict[index] = code

        if ent is None:
            # No targets?
            LOGGER.warning(
                'No entities found with name "{}", for '
                'comp_scriptvar at {}!',
                comp_ent['target'], comp_ent['origin'],
            )

    for ent, var_dict in set_vars.items():
        full_code = []
        for var_name, values in var_dict.items():



# Functions to call to compute the data to read.
def mode_const(comp_ent: Entity, ent: Entity) -> str:
    """Set a simple constant."""
    return comp_ent['const']


def mode_name(comp_ent: Entity, ent: Entity) -> str:
    """Set the value to the entity name."""
    return '"{}"'.format(ent['targetname'])


def mode_handle(comp_ent: Entity, ent: Entity) -> str:
    """Compute and return a handle to tis entity."""
    if ent['targetname']:
        return 'Entities.FindByName(null, "{}")'.format(ent['targetname'])
    else:
        # No name, use classname and position.
        return 'Entities.FindByClassnameWithin(null, "{}", {}, 1)'.format(
            ent['classname'],
            vs_vec(Vec.from_str(ent['origin']))
        )


def mode_pos(comp_ent: Entity, ent: Entity) -> str:
    """Return the position of the entity."""
    pos = Vec.from_str(ent['origin'])
    scale = conv_float(comp_ent['const'], 1.0)
    return vs_vec(scale * pos)


def mode_ang(comp_ent: Entity, ent: Entity) -> str:
    """Return the angle of the entity, as a Vector."""
    return vs_vec(Vec.from_str(ent['angles']))


def mode_off(comp_ent: Entity, ent: Entity) -> str:
    """Return the offset from the ent to the reference."""
    scale = conv_float(comp_ent['const'], 1.0)
    offset = Vec.from_str(ent['origin']) - Vec.from_str(comp_ent['origin'])
    return vs_vec(offset * scale)


def _mode_axes(norm: Vec) -> Callable[[Entity, Entity], str]:
    """Return the given direction vector."""
    def mode_func(comp_ent: Entity, ent: Entity) -> str:
        """Rotate the axis by the given value."""
        out = norm.copy().rotate_by_str(ent['angles', '0 0 0'])
        scale = conv_float(comp_ent['const'], 1.0)
        return vs_vec(scale * out)
    return mode_func


mode_x = _mode_axes(Vec(x=1.0))
mode_y = _mode_axes(Vec(y=1.0))
mode_z = _mode_axes(Vec(z=1.0))
