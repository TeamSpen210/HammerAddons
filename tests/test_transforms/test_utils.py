"""Test utilty transform logic."""
import logging
from decimal import Decimal
import operator

from srctools import VMF, Entity
import pytest

from hammeraddons.bsp_transform.common import (
    check_control_enabled, get_multimode_value,
    parse_numeric_specifier, op_always_fail,
)


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


def test_numeric_parsing_good(caplog: pytest.LogCaptureFixture) -> None:
    """Test the numeric parsing function, with good values."""
    caplog.set_level(logging.WARNING)
    assert parse_numeric_specifier('  42 ') == (operator.eq, Decimal(42))
    assert parse_numeric_specifier('<3') == (operator.lt, Decimal(3))
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


def test_numeric_parsing_bad_op(caplog: pytest.LogCaptureFixture) -> None:
    """Test the numeric parsing function, when given an illegal operator."""
    assert parse_numeric_specifier(' <> 34.83') == (operator.eq, Decimal('34.83'))
    assert parse_numeric_specifier(' >!> 34.83') == (operator.eq, Decimal('34.83'))
    assert caplog.record_tuples == [
        ('srctools.hammeraddons.bsp_transform.common', logging.WARNING, 'Invalid numeric operator "<>"'),
        ('srctools.hammeraddons.bsp_transform.common', logging.WARNING, 'Invalid numeric operator ">!>"'),
    ]


def test_numeric_parsing_bad_number(caplog: pytest.LogCaptureFixture) -> None:
    """Test the numeric parsing function, when given a totally invalid number."""
    assert parse_numeric_specifier('hello') == (op_always_fail, Decimal())
    assert caplog.record_tuples == [
        ('srctools.hammeraddons.bsp_transform.common', logging.WARNING, 'Invalid number "hello"')
    ]


@pytest.mark.parametrize('mode, result', [
    ('local', 'a_local'),
    ('global', 'a_global'),
    ('pos', '4 8 12'),
    ('position', '4 8 12'),
])
def test_multimode_value(caplog: pytest.LogCaptureFixture, mode: str, result: str) -> None:
    """Test regular multimode values."""
    assert get_multimode_value(
        Entity(VMF(), {
            'value_mode_01': mode,
        }),
        prefix='value_',
        suffix='_01',
        desc='test',
    ) == ''

    assert get_multimode_value(
        Entity(VMF(), {
            'value_mode_01': mode,
            'value_local_01': 'a_local',
            'value_global_01': 'a_global',
            'value_pos_01': '4 8 12'
        }),
        prefix='value_',
        suffix='_01',
        desc='test',
    ) == result
    assert caplog.record_tuples == []


def test_multimode_legacy(caplog: pytest.LogCaptureFixture) -> None:
    """Test handling legacy multimode values."""
    vmf = VMF()
    assert get_multimode_value(
        Entity(vmf, {
            'classname': 'info_target',
            'value_local_01': 'a_local',
            'value_global_01': 'a_global',
        }),
        prefix='value_',
        suffix='_01',
        desc='testing',
    ) == 'a_global'

    assert get_multimode_value(
        Entity(vmf, {
            'classname': 'info_target',
            'value_local_01': 'a_local',
        }),
        prefix='value_',
        suffix='_01',
        desc='testing',
    ) == 'a_local'
    assert caplog.record_tuples == []


def test_multimode_invalid(caplog: pytest.LogCaptureFixture) -> None:
    """Test invalid multimode values."""
    assert get_multimode_value(
        Entity(VMF(), {
            'classname': 'info_target',
            'origin': '1 2 3',
            'value_local_01': 'a_local',
            'value_mode_01': 'not_a_mode',
        }),
        prefix='value_',
        suffix='_01',
        desc='testing',
    ) == 'a_local'
    assert caplog.record_tuples == [
        (
            'srctools.hammeraddons.bsp_transform.common',
            logging.WARNING,
            'Invalid testing mode "not_a_mode" for info_target @ (1 2 3)!',
        ),
    ]
