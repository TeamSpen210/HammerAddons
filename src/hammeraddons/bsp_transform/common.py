"""Operations that can be reused across different transforms."""
import operator
import re
from typing import Callable, Container, Dict, Iterator, Tuple, Union, final
from decimal import Decimal, InvalidOperation

from typing_extensions import Literal, Self, TypeAlias
import attrs

from srctools import Entity, FrozenVec, VMF, Vec, conv_bool
from srctools.fgd import EntityDef
from srctools.logger import get_logger


__all__ = [
    'parse_numeric_specifier', 'check_control_enabled',
    'ent_description', 'RelayOut', 'get_multimode_value', 'strip_cust_keys'
]


LOGGER = get_logger(__name__)
NumericOp: TypeAlias = Callable[[Decimal, Decimal], bool]
NumericSpecifier: TypeAlias = Tuple[NumericOp, Decimal]

OPERATIONS: Dict[str, NumericOp] = {
    '<': operator.lt,
    '>': operator.gt,
    '>=': operator.ge,
    '<=': operator.le,
    '=': operator.eq,
    '==': operator.eq,
    '!=': operator.ne,
    '=!=': operator.ne,
    '~=': operator.ne,
    '=/=': operator.ne,
}
# Matches multiple characters present in OPERATIONS.
# \s skips whitespace beforehand, so we have a capturing group to just grap the actual operation.
OPERATION_RE = re.compile(r'\s*([{}]+)'.format(''.join(map(re.escape, {
    char for key in OPERATIONS for char in key
}))))
kv_name_cache: Dict[str, Container[str]] = {}


# noinspection PyUnusedLocal
def op_always_fail(a: Decimal, b: Decimal, /) -> Literal[False]:
    """NumericOp implementation which always fails."""
    return False


def parse_numeric_specifier(text: str, desc: str='') -> NumericSpecifier:
    """Parse case values like "> 5" into the operation and number."""
    operation: NumericOp
    if (match := OPERATION_RE.match(text)) is not None:
        try:
            operation = OPERATIONS[match.group(1)]
        except KeyError:
            LOGGER.warning('Invalid numeric operator "{}"{}', match.group(1), desc)
            operation = operator.eq
        num_str = text[match.end():]
    else:
        operation = operator.eq
        num_str = text
    try:
        num = Decimal(num_str)
    except InvalidOperation:
        LOGGER.warning('Invalid number "{}"{}', num_str, desc)
        # Force this to always fail.
        return (op_always_fail, Decimal())
    else:
        return (operation, num)


def check_control_enabled(ent: Entity) -> bool:
    """Implement the bahaviour of ControlEnables - control_type and control_value.

    This allows providing a fixup value, and optionally inverting it.
    """
    # If ctrl_type is 0, ctrl_value needs to be 1 to be enabled.
    # If ctrl_type is 1, ctrl_value needs to be 0 to be enabled.
    if 'ctrl_type' in ent:
        return conv_bool(ent['ctrl_type'], False) != conv_bool(ent['ctrl_value'], True)
    else:
        # Missing, assume true if ctrl_value also isn't present.
        return conv_bool(ent['ctrl_value'], True)


def ent_description(ent: Entity) -> str:
    """Return an identifiable description for an entity."""
    name = ent['targetname']
    classname = ent['classname']
    pos = ent['origin']
    if name:
        return f'"{name}" {classname} @ ({pos})'
    else:
        return f'{classname} @ ({pos})'


@final
@attrs.frozen
class RelayOut:
    """Entity name, plus the relay input/output to use."""
    ent: Entity
    input: str
    output: str

    @classmethod
    def create(cls, vmf: VMF, pos: Union[Vec, FrozenVec], name: str) -> Iterator[Self]:
        """Generates a valid entity along with a free input/output pair."""
        # Could also use func_instance_io_proxy, but only in L4D+, and it might be weird.
        user_outs = [('Trigger', 'OnTrigger')] + [
            (f'FireUser{x}', f'OnUser{x}')
            for x in range(1, 5)
        ]
        while True:
            ent = vmf.create_ent(
                'logic_relay',
                origin=pos,
                spawnflags='2',  # Allow fast retrigger
            ).make_unique(name)
            for inp, out in user_outs:
                yield cls(ent, inp, out)


def get_multimode_value(ent: Entity, *, prefix: str='', suffix: str='', desc: str) -> str:
    """Read from differerent typed keyvalues, specified by a mode option.

    The mode was originally not present, which is why local/global is doubled up.
    """
    mode = ent[f'{prefix}mode{suffix}', 'legacy'].casefold()
    if mode == 'legacy':
        return ent[f'{prefix}global{suffix}'] or ent[f'{prefix}local{suffix}']
    elif mode == 'global':
        return ent[f'{prefix}global{suffix}']
    elif mode == 'local':
        return ent[f'{prefix}local{suffix}']
    elif mode in ['pos', 'position']:
        return ent[f'{prefix}pos{suffix}']
    else:
        LOGGER.warning(
            'Invalid {} mode "{}" for {}!',
            desc, mode, ent_description(ent),
        )
        return ent[f'{prefix}global{suffix}'] or ent[f'{prefix}local{suffix}']


def strip_cust_keys(ent: Entity) -> None:
    """Strip all keyvalues from this entity that are not valid.

    This is useful for postcompiler ents that get converted to regular ones.
    """
    classname = ent['classname'].casefold()
    try:
        names = kv_name_cache[classname]
    except KeyError:
        try:
            ent_def = EntityDef.engine_def(classname)
        except KeyError:
            LOGGER.warning('Unknown classname "{}"!', classname)
            names = ()
        else:
            names = set(ent_def.kv)
        kv_name_cache[classname] = names
    for key in list(ent):
        if key not in names:
            del ent[key]
