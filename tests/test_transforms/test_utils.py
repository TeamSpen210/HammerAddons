"""Test utilty transform logic."""
import logging
from decimal import Decimal
import operator

from srctools import VMF, Entity

from hammeraddons.bsp_transform.common import check_control_enabled, parse_numeric_specifier, op_always_fail


def test_check_control_start_enabled() -> None:
    """Test the control-enabled options functions, for the start-enabled mode."""
    vmf = VMF()
    assert not check_control_enabled(Entity(vmf, {
        'ctrl_type': '0',  # Start enabled.
        'ctrl_value': '0',
    }))
    assert check_control_enabled(Entity(vmf, {
        'ctrl_type': '0',
        'ctrl_value': '1',
    }))


def test_check_control_start_disabled() -> None:
    """Test the control-enabled options functions, for the start-disabled mode."""
    vmf = VMF()
    assert check_control_enabled(Entity(vmf, {
        'ctrl_type': '1',  # Start disabled
        'ctrl_value': '0',
    }))
    assert not check_control_enabled(Entity(vmf, {
        'ctrl_type': '1',
        'ctrl_value': '1',
    }))


def test_check_control_backward_compatiblity() -> None:
    """Test the control-enabled options functions, when the mode is missing."""
    vmf = VMF()
    # Backwards compatibility, if ctrl_type is missing assume Start Enabled.
    assert not check_control_enabled(Entity(vmf, {
        'ctrl_value': '0',
    }))
    assert check_control_enabled(Entity(vmf, {
        'ctrl_value': '1',
    }))


def test_numeric_parsing_good(caplog) -> None:
    """Test the numeric parsing function, with good values."""
    caplog.set_level(logging.WARNING)
    assert parse_numeric_specifier('  42 ') == (operator.eq, Decimal('42'))
    assert parse_numeric_specifier('<3') == (operator.lt, Decimal('3'))
    assert parse_numeric_specifier(' > 45.872_67') == (operator.gt, Decimal('45.87267'))
    assert parse_numeric_specifier('>= -0892.930') == (operator.ge, Decimal('-892.93'))
    assert parse_numeric_specifier('=4') == (operator.eq, Decimal(4))
    assert parse_numeric_specifier('== -17') == (operator.eq, Decimal(-17))
    assert parse_numeric_specifier(' != 909') == (operator.ne, Decimal(909))
    assert parse_numeric_specifier(' =!= 87') == (operator.ne, Decimal(87))
    assert parse_numeric_specifier('~= 38') == (operator.ne, Decimal(38))
    assert parse_numeric_specifier(' =/= 24.5') == (operator.ne, Decimal('24.5'))
    # No warnings caught.
    assert caplog.record_tuples == []


def test_numeric_parsing_bad_op(caplog) -> None:
    """Test the numeric parsing function, when given an illegal operator."""
    assert parse_numeric_specifier(' <> 34.83') == (operator.eq, Decimal('34.83'))
    assert parse_numeric_specifier(' >!> 34.83') == (operator.eq, Decimal('34.83'))
    assert caplog.record_tuples == [
        ('srctools.hammeraddons.bsp_transform.common', logging.WARNING, 'Invalid numeric operator "<>"'),
        ('srctools.hammeraddons.bsp_transform.common', logging.WARNING, 'Invalid numeric operator ">!>"'),
    ]


def test_numeric_parsing_bad_number(caplog) -> None:
    """Test the numeric parsing function, when given a totally invalid number."""
    assert parse_numeric_specifier('hello') == (op_always_fail, Decimal())
    assert caplog.record_tuples == [
        ('srctools.hammeraddons.bsp_transform.common', logging.WARNING, 'Invalid number "hello"')
    ]
