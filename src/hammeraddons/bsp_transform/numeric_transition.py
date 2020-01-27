"""Implements an entity which transitions a value from one to another."""
import math
from string import ascii_lowercase
from typing import Callable, Dict, List, Optional

from srctools import conv_float, Output
from srctools.bsp_transform import trans, Context
from srctools.logger import get_logger

LOGGER = get_logger(__name__)

halfpi = math.pi / 2.0

# Offsets to use when computing the derivative.
EPSILON = 1e-6

BRIGHT_LETTERS = {
    let : i/25
    for i, let in enumerate(ascii_lowercase)
}

TRANSITION_KVS = [
    "beat_interval", "delay", "duration",
    "easing_end", "easing_start",
    "endval",
    "io_type",
    "line_trans2", "line_trans3",
    "opt_name", "startval",
    "target", "transform",
]


def lerp(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Linearly interpolate from (in_min, in_max) -> (out_min, out_max)."""
    return out_min + (
        ((x - in_min) * (out_max - out_min)) /
        (in_max - in_min)
    )


@trans('comp_numeric_transition')
def numeric_transition(ctx: Context) -> None:
    """When triggered, animates a keyvalue/input over time with various options."""
    for ent in ctx.vmf.by_class['comp_numeric_transition']:
        ent['classname'] = 'logic_relay'

        target = ent['target']
        if not target:
            LOGGER.warning('Numeric transition at {} has no target!', ent['origin'])
            continue

        beat_time = conv_float(ent['beat_interval'], 0.1)
        abs_delay = conv_float(ent['delay'])
        duration = conv_float(ent['duration'], 5.0)
        # Determines how the raw position is transformed into a function.
        transform_type = ent['transform', 'set'].casefold()

        # Special case - if the transform type is "light", allow parsing these
        # as A-Z values.
        value_start = value_end = None  # type: Optional[float]
        if transform_type == 'light':
            value_start = BRIGHT_LETTERS.get(ent['startval'].lower(), None)
            value_end = BRIGHT_LETTERS.get(ent['endval'].lower(), None)

        if value_start is None:
            value_start = conv_float(ent['startval'], 0)
        if value_end is None:
            value_end = conv_float(ent['endval'], 100.0)

        input_name = ent['opt_name', 'SetSpeed']
        io_type = ent['io_type', 'auto'].casefold()
        if io_type not in ('io', 'kv'):
            # Find the target, to determine whether this is a KV or IO.
            try:
                targ_ent = next(ctx.vmf.search(target))
            except StopIteration:
                LOGGER.warning(
                    'No target "{}" found for '
                    'numeric transition "{}"! You must manually specify '
                    'if it\'s a KV or IO!',
                    target,
                    ent['targetname'],
                )
                continue
            try:
                targ_cls = ctx.fgd.entities[targ_ent['classname']]
            except KeyError:
                LOGGER.warning(
                    'Unknown classname {} '
                    'for entity "{}"!',
                    targ_ent['classname'], target,
                )
                continue
            if input_name in targ_cls.inp:
                io_type = 'io'
            elif input_name in targ_cls.kv:
                io_type = 'kv'
            else:
                LOGGER.warning(
                    'Classname {} has no keyvalue or input {}'
                    'You must manually specify '
                    'if it\'s a KV or IO for numeric transition "{}"!',
                    targ_cls.classname, input_name,
                    ent['targetname'],
                )
                continue

        ease_start_type = ent['easing_start', 'linear'].casefold()
        ease_end_type = ent['easing_end', 'linear'].casefold()

        # We've parsed all the names, so strip those keyvalues.
        for keyvalue in TRANSITION_KVS:
            del ent[keyvalue]

        # Now, lookup the function used for the easing types.
        try:
            ease_start_func = EASE_START_FUNC[ease_start_type]
        except KeyError:
            LOGGER.warning('Unknown easing start type "{}" for "{}"', target)
            ease_start_func = ease_func_linear

        try:
            ease_end_func = EASE_START_FUNC[ease_end_type]
        except KeyError:
            LOGGER.warning('Unknown easing start type "{}" for "{}"', target)
            ease_end_func = ease_func_linear

        def compute_point(x: float) -> float:
            pos = x * ease_start_func(x) + (1.0 - x) * ease_end_func(x)
            if pos < 0.0:
                pos = 0.0
            if pos > 1.0:
                pos = 1.0
            return lerp(pos, 0, 1, value_start, value_end)

        point_count = math.ceil(duration / beat_time)
        if point_count <= 0:
            LOGGER.warning('Numeric transition "{}" has no duration at all?', ent['targetname'])
            point_count = 1
        points = [i / point_count for i in range(int(point_count))]

        if transform_type == 'speed':
            # Compute the speed from x to x+1
            result = (
                (compute_point(x) + compute_point(x + 1/point_count)) / beat_time
                for x in points
            )
        elif transform_type == 'moveto':
            # The input makes the object move to a position over time.
            # So we want to use the point after the current, so it should
            # reach that about when we next fire an output.
            result = (
                compute_point(x + 1/point_count)
                for x in points
            )
        elif transform_type == 'light':
            # Generate light pattern names from the point.
            result = (
                ascii_lowercase[max(0, min(25, int(round(compute_point(x) * 25))))]
                for x in points
            )
        else:
            if transform_type != 'set':
                LOGGER.warning(
                    '"{}" is not a valid transform type '
                    'for numeric transition "{}"!',
                    transform_type,
                    ent['targetname'],
                )
            # Directly sets the location at every point.
            result = map(compute_point, points)

        last_inp = None
        for i, point in enumerate(result):
            if io_type == 'kv':
                io_input = 'AddOutput'
                param = '{} {}'.format(input_name, point)
            else:  # input
                io_input = input_name
                param = format(point)

            if param != last_inp:
                ent.add_out(Output(
                    "OnTrigger",
                    target,
                    io_input, param,
                    delay=abs_delay + beat_time * i,
                ))
                last_inp = param


# Functions to produce the desired curves.
# Given an input from 0-1, produces an output from 0-1.


def ease_func_linear(x: float) -> float:
    """Do no alteration to the value."""
    return x


def ease_func_power_start(power: int):
    """Generate the polynomial easing in functions."""
    def func_start(x: float) -> float:
        """Apply a polynomial easing in."""
        return x ** power
    return func_start


def ease_func_power_end(power: int):
    """Generate the polynomial easing out functions."""
    def func_end(x: float) -> float:
        """The function for a specific power."""
        return 1.0 - (1.0 - x) ** power
    return func_end


def ease_func_sine_start(x: float) -> float:
    """Apply a sinusoidal transform."""
    return math.sin(x * halfpi)


def ease_func_sine_end(x: float) -> float:
    """Apply a sinusoidal transform."""
    return 1.0 - math.cos(x * halfpi)


EASE_START_FUNC = {
    'linear': ease_func_linear,
    'quad': ease_func_power_start(2),
    'cubic': ease_func_power_start(3),
    'quartic': ease_func_power_start(4),
    'sine': ease_func_sine_start,
}  # type: Dict[str, Callable[[float], float]]

EASE_END_FUNC = {
    'linear': ease_func_linear,
    'quad': ease_func_power_end(2),
    'cubic': ease_func_power_end(3),
    'quartic': ease_func_power_end(4),
    'sine': ease_func_sine_end,
}  # type: Dict[str, Callable[[float], float]]
