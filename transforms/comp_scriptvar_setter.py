"""Implements comp_scriptvar_setter."""
import re
from collections import defaultdict
from typing import Dict, Type, Optional, Callable, Union

from srctools.bsp_transform import trans, Context
from srctools.fgd import FGD, ValueTypes
from srctools.logger import get_logger
from srctools.tokenizer import escape_text
from srctools import Entity, Vec, conv_float, conv_bool, Angle


LOGGER = get_logger(__name__)
MODES: Dict[str, Callable[[Entity, Entity, FGD], str]] = {}


def vs_vec(vec: Vec) -> str:
    """Convert the provided Vec into a VScript Vector constructor code."""
    return 'Vector({})'.format(vec.join())


def squirrel_string(val: str) -> str:
    """Wrap the value in double-quotes to make Squirrel code to construct the string."""
    return f'"{escape_text(val)}"'


class VarData:
    """The info stored on a variable."""
    def __init__(self) -> None:
        # Non-array values.
        self.scalar: Optional[str] = None
        # Array values at a specific index.
        self.specified_pos: dict[int, str] = {}
        # Array values at anywhere that fits.
        self.extra_pos: set[str] = set()

    @property
    def is_array(self) -> bool:
        return bool(self.specified_pos or self.extra_pos)

    def make_code(self) -> str:
        """Generate the code for setting this."""
        if self.is_array:
            # First build an array big enough to fit everything.
            array: list[Optional[str]] = [None] * (
                max(self.specified_pos.keys(), default=0) + 1 +
                len(self.extra_pos)
            )

            for ind, value in self.specified_pos.items():
                array[ind] = value

            # Then insert each extra value in at the first space that fits.
            # Use the start parameter to skip values we know are None.
            ind = 0
            for value in sorted(self.extra_pos):
                ind = array.index(None, ind)
                array[ind] = value

            # Strip any overallocated Nones at the end.
            while array[-1] is None:
                array.pop()

            return '[\n{}\n]'.format(', \n'.join([
                '\t null' if x is None else '\t' + x
                for x in array
            ]))
        else:
            return self.scalar


@trans('comp_scriptvar_setter')
def comp_scriptvar(ctx: Context):
    """An entity to allow setting VScript variables to information from the map."""
    # {ent: {variable: data}}
    set_vars: dict[Entity, dict[str, VarData]] = defaultdict(lambda: defaultdict(VarData))
    # If the index is None, there's no index.
    # If an int, that specific one.
    # If ..., blank index and it's inserted anywhere that fits.

    for comp_ent in ctx.vmf.by_class['comp_scriptvar_setter']:
        comp_ent.remove()
        var_name = comp_ent['variable']
        index: Union[int, Type[Ellipsis], None] = None

        parsed_match = re.fullmatch(r'\s*([^[]+)\[([0-9]*)]\s*', var_name)
        if parsed_match:
            var_name, index_str = parsed_match.groups()
            if index_str:
                try:
                    index = int(index_str)
                    if index < 0:
                        raise ValueError  # No negatives.
                except (TypeError, ValueError):
                    LOGGER.warning(
                        'Invalid variable index in '
                        'comp_scriptvar at {} targetting "{}"!',
                        comp_ent['origin'], comp_ent['target']
                    )
                    continue
            else:
                index = ...
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
            mode_func = MODES[comp_ent['mode']]
        except KeyError:
            LOGGER.warning(
                'Invalid mode "{}" in '
                'comp_scriptvar at {} targeting "{}"!',
                comp_ent['mode'], comp_ent['origin'], comp_ent['target'],
            )
            continue
        else:
            code = mode_func(comp_ent, ref_ent, ctx.fgd)

        ent: Optional[Entity] = None
        for ent in ctx.vmf.search(comp_ent['target']):
            var_data = set_vars[ent][var_name]
            # Now we've got to match the assignment this is doing
            # with other scriptvars.

            # First, take care of scalar assignment.
            if index is None:
                if var_data.is_array:
                    LOGGER.warning(
                        "comp_scriptvar at {} can't set a non-array value "
                        'on top of the array {} on the entity "{}"!',
                        comp_ent['origin'], var_name, comp_ent['target'],
                    )
                elif var_data.scalar is not None:
                    LOGGER.warning(
                        'comp_scriptvar at {} overwrote '
                        'the variable {} on "{}"!',
                        comp_ent['origin'], var_name, comp_ent['target'],
                    )
                else:
                    var_data.scalar = code
                continue
            # Else, we're setting an array value.
            if var_data.scalar is not None:
                LOGGER.warning(
                    "comp_scriptvar at {} can't set an array value "
                    'on top of the non-array {} on the entity "{}"!',
                    comp_ent['origin'], var_name, comp_ent['target'],
                )
                continue
            if index is Ellipsis:
                var_data.extra_pos.add(code)
            else:
                # Allow duplicates that write the exact same thing,
                # as a special case.
                if var_data.specified_pos.get(index, code) != code:
                    LOGGER.warning(
                        "comp_scriptvar at {} can't "
                        'overwrite {}[{}] on the entity "{}"!',
                        comp_ent['origin'], var_name, index, comp_ent['target'],
                    )
                else:
                    var_data.specified_pos[index] = code

        if ent is None:
            # No targets?
            LOGGER.warning(
                'No entities found with name "{}", for '
                'comp_scriptvar at {}!',
                comp_ent['target'], comp_ent['origin'],
            )

    for ent, var_dict in set_vars.items():
        full_code = []
        for var_name, var_data in var_dict.items():
            full_code.append('{} <- {};'.format(
                var_name, var_data.make_code()
            ))
        if full_code:
            ctx.add_code(ent, '\n'.join(full_code))


# Functions to call to compute the data to read.
def mode_const(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
    """Set a simple constant."""
    return comp_ent['const']
    

def mode_string(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
    """Set a constant, as a string."""
    return '"{}"'.format(comp_ent['const'])


def mode_bool(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
    """Convert the value to a boolean."""
    return 'true' if conv_bool(comp_ent['const']) else 'false'


def mode_inv_bool(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
    """Convert the value to a boolean, and invert it."""
    return 'false' if conv_bool(comp_ent['const']) else 'true'


def mode_name(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
    """Set the value to the entity name."""
    return '"{}"'.format(ent['targetname'])


def mode_handle(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
    """Compute and return a handle to this entity."""
    if ent['targetname']:
        return 'Entities.FindByName(null, "{}")'.format(ent['targetname'])
    else:
        # No name, use classname and position.
        return 'Entities.FindByClassnameWithin(null, "{}", {}, 1)'.format(
            ent['classname'],
            vs_vec(Vec.from_str(ent['origin']))
        )


def mode_keyvalue(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
    """Read a keyvalue from the entity, then match the type to a Squirrel type."""
    if ent is comp_ent:
        LOGGER.warning(
            'No reference entity for keyvalue-mode '
            'comp_scriptvar at {} targeting "{}"!',
            comp_ent['origin'], comp_ent['target'],
        )
        return 'null'
    keyvalue = comp_ent['const']
    key = ent[comp_ent['const']]
    try:
        key_info = fgd[ent['classname']].kv[keyvalue]
    except KeyError:
        LOGGER.warning(
            'No definition for keyvalue "{}" for {} entities: '
            'comp_scriptvar at {} targeting "{}"!\nAssuming it\'s a string.',
            keyvalue, ent['classname'],
            comp_ent['origin'], comp_ent['target'],
        )
        return f'"{key}"'
    if not key:
        key = key_info.default
    return KEYVALUES.get(key_info.type, squirrel_string)(key)


def mode_pos(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
    """Return the position of the entity."""
    pos = Vec.from_str(ent['origin'])
    scale = conv_float(comp_ent['const'], 1.0)
    return vs_vec(scale * pos)


def mode_ang(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
    """Return the angle of the entity, as a Vector."""
    return vs_vec(Vec.from_str(ent['angles']))


def mode_off(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
    """Return the offset from the ent to the reference."""
    scale = conv_float(comp_ent['const'], 1.0)
    offset = Vec.from_str(ent['origin']) - Vec.from_str(comp_ent['origin'])
    return vs_vec(offset * scale)


def mode_dist(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
    """Return the distance from the ent to the reference."""
    scale = conv_float(comp_ent['const'], 1.0)
    offset = Vec.from_str(ent['origin']) - Vec.from_str(comp_ent['origin'])
    return str(offset.mag() * scale)


def _mode_axes(norm: Vec) -> Callable[[Entity, Entity, FGD], str]:
    """Return the given direction vector."""
    def mode_func(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
        """Rotate the axis by the given value."""
        out = round(norm @ Angle.from_str(ent['angles', '0 0 0']), 6)
        scale = conv_float(comp_ent['const'], 1.0)
        return vs_vec(scale * out)
    return mode_func
    

def _mode_pos_axis(axis: str) -> Callable[[Entity, Entity, FGD], str]:
    """Return a single axis of the ent's position."""
    def mode_func(comp_ent: Entity, ent: Entity, fgd: FGD) -> str:
        """Rotate the axis by the given value."""
        pos = Vec.from_str(ent['origin'])
        scale = conv_float(comp_ent['const'], 1.0)
        return str(pos[axis] * scale)
    return mode_func


MODES['x'] = _mode_axes(Vec(x=1.0))
MODES['y'] = _mode_axes(Vec(y=1.0))
MODES['z'] = _mode_axes(Vec(z=1.0))
MODES['pos_x'] = _mode_pos_axis('x')
MODES['pos_y'] = _mode_pos_axis('y')
MODES['pos_z'] = _mode_pos_axis('z')
MODES.update(
    (name[5:], func)
    for name, func in globals().items()
    if name.startswith('mode_')
)

# Keyvalue types -> equivalent Squirrel code, if not just stringified.
KEYVALUES = {
    ValueTypes.VOID: lambda val: 'null',
    ValueTypes.SPAWNFLAGS: str,

    # Simple values
    ValueTypes.BOOL: lambda v: 'true' if conv_bool(v) else 'false',
    ValueTypes.INT: str,
    ValueTypes.FLOAT: str,
    ValueTypes.VEC: lambda val: vs_vec(Vec.from_str(val)),
    ValueTypes.ANGLES: lambda val: vs_vec(Vec.from_str(val)),

    ValueTypes.STR_VSCRIPT: lambda val: f'[{", ".join(map(squirrel_string, val.split()))}]',

    ValueTypes.VEC_LINE: lambda val: vs_vec(Vec.from_str(val)),
    ValueTypes.VEC_ORIGIN: lambda val: vs_vec(Vec.from_str(val)),
    ValueTypes.VEC_AXIS: lambda val: vs_vec(Vec.from_str(val)),

    # Space seperated, convert to an array.
    ValueTypes.COLOR_1: lambda val: f'[{", ".join(val.split())}]',
    ValueTypes.COLOR_255: lambda val: f'[{", ".join(val.split())}]',
    ValueTypes.SIDE_LIST: lambda val: f'[{", ".join(val.split())}]',

    ValueTypes.EXT_VEC_DIRECTION: lambda val: vs_vec(Vec.from_str(val)),
    ValueTypes.EXT_VEC_LOCAL: lambda val: vs_vec(Vec.from_str(val)),
    ValueTypes.EXT_ANGLE_PITCH: lambda val: vs_vec(Vec.from_str(val)),
    ValueTypes.EXT_ANGLES_LOCAL: lambda val: vs_vec(Vec.from_str(val)),
}
