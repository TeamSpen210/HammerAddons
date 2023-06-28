"""Operations that can be reused across different transforms."""
import operator
import re
from typing import Callable, Tuple
from decimal import Decimal

from typing_extensions import TypeAlias

from srctools import Entity, conv_bool
from srctools.logger import get_logger


LOGGER = get_logger(__name__)
NumericOp: TypeAlias = Callable[[Decimal, Decimal], bool]
NumericSpecifier: TypeAlias = Tuple[NumericOp, Decimal]

OPERATIONS: dict[str, NumericOp] = {
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
# Matches characters present in OPERATIONS
OPERATION_RE = re.compile('({0})+'.format('|'.join(map(re.escape, {
    char for key in OPERATIONS for char in key
}))))


def parse_numeric_specifier(text: str, desc: str) -> NumericSpecifier:
    """Parse case values like "> 5" into the operation and number."""
    operation: NumericOp
    if (match := OPERATION_RE.match(text)) is not None:
        try:
            operation = OPERATIONS[match.group()]
        except KeyError:
            LOGGER.warning('Invalid numeric operator "{}" {}', match.group(), desc)
            operation = operator.eq
        num_str = text[match.end():]
    else:
        operation = operator.eq
        num_str = text
    try:
        num = Decimal(num_str)
    except ValueError:
        LOGGER.warning('Invalid number "{}" {}', num_str, desc)
        # Force this to always fail.
        return (lambda a, b: False, Decimal())
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
