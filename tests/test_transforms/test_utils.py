"""Test utilty transform logic."""
from srctools import VMF, Entity

from hammeraddons.bsp_transform import check_control_enabled


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
