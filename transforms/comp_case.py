"""comp_case is a compile-time collapsible version of logic_case."""
from typing import Dict, Iterator, List, Tuple

import hashlib
import re
import struct
import random
from decimal import Decimal
from collections import defaultdict

from srctools import Entity, Output, conv_bool
from srctools.math import parse_vec_str
from srctools.logger import get_logger

from hammeraddons.bsp_transform import (
    trans, Context,
    check_control_enabled,
    NumericSpecifier, parse_numeric_specifier,
)


LOGGER = get_logger(__name__)
CASES = [f'case{x:02}' for x in range(1, 17)]


def collapse_case(ctx: Context, case: Entity) -> None:
    """Collapse a single case."""
    multi_cases = conv_bool(case['multiplecasesallowed'])
    case_name = case['targetname']
    mode = case['mode'].casefold()
    default_value = case['value']

    desc = f'for comp_case "{case_name}" @ ({case["origin"]})'

    hasher_template = hashlib.sha512()
    hasher_template.update(f"{case['seed']};{case_name}".encode('utf-8'))
    hasher_template.update(struct.pack('<x3f', *parse_vec_str(case['origin'])))

    # Find all defined outputs and parameters, so we can loop through them.
    out_cases: Dict[int, List[Output]] = defaultdict(list)
    out_default: List[Output] = []
    out_used: List[Output] = []
    out_matched: List[Output] = []
    for out in case.outputs:
        if out.output.casefold().startswith('oncase'):
            try:
                num = int(out.output[6:])
            except ValueError:
                LOGGER.warning('Unknown case output "{}" {}',out.output, desc)
                continue
            out_cases[num].append(out)
        elif out.output.casefold() == 'ondefault':
            out_default.append(out)
        elif out.output.casefold() == 'onused':
            out_used.append(out)
        elif out.output.casefold() == 'onmatched':
            out_matched.append(out)

    case_params: Dict[int, str] = {}
    for k, v in case.items():
        if k.casefold().startswith('case'):
            try:
                num = int(k[4:])
            except ValueError:
                LOGGER.warning('Unknown case keyvalue "{}" {}', k, desc)
                continue
            case_params[num] = v

    # Sort the keys, so we check in order.
    key_out = sorted(out_cases)
    key_params = sorted(case_params)

    if mode == 'string':
        def find_matches(param: str) -> Iterator[int]:
            """Find string-based matches."""
            for case_num in key_params:
                if param == case_params[case_num]:
                    yield case_num
    elif mode == 'casefold':
        def find_matches(param: str) -> Iterator[int]:
            """Find string-based matches, ignoring cases."""
            for case_num in key_params:
                if param.casefold() == case_params[case_num].casefold():
                    yield case_num
    elif mode == 'numeric':
        # Pre-parse "< 5" style values.
        numeric_cases: List[Tuple[NumericSpecifier, int]] = [
            (parse_numeric_specifier(
                case_params[case_num],
                f' for case #{case_num} in {desc}'
            ), case_num)
            for case_num in key_params
        ]

        def find_matches(param: str) -> Iterator[int]:
            """Find numeric matches, using a configurable operator."""
            try:
                num_a = Decimal(param)
            except ValueError:
                return  # Matches nothing.
            for (operation, num_b), case_num in numeric_cases:
                if operation(num_a, num_b):
                    yield case_num
    else:
        LOGGER.error(
            'Invalid mode "{}" for comp_case "{}" @ ({})',
            case['mode'], case_name, case['origin']
        )
        return

    def compute_outputs(param: str) -> Iterator[Output]:
        """Compute the matching cases, then yield the outputs."""
        matching = find_matches(param)
        yield from out_used  # Always used.
        try:
            first_match = next(matching)
        except StopIteration:
            # No match, use defaults.
            yield from out_default
            return
        yield from out_cases[first_match]
        yield from out_matched
        if multi_cases:  # Include all matches.
            for match in matching:
                yield from out_cases[match]

    def handle_pick_random(source: Entity, out: Output) -> List[Output]:
        """Handle the PickRandom input."""
        hasher = hasher_template.copy()
        hasher.update((source['targetname'] or source['classname']).encode('utf8'))
        hasher.update(struct.pack('<x3f', *parse_vec_str(source['origin'])))
        rng = random.Random(hasher.digest())
        return out_cases[rng.choice(key_out)]

    ctx.add_io_remap_func(
        case_name, 'InValue',
        lambda source, out: list(compute_outputs(out.params or default_value)),
    )
    ctx.add_io_remap_func(
        case_name, 'Trigger',
        lambda source, out: list(compute_outputs(default_value)),
    )
    ctx.add_io_remap_func(case_name, 'PickRandom', handle_pick_random)


@trans('comp_case', priority=10)
def comp_case(ctx: Context) -> None:
    """A version of logic_case which is collapsed at compile time."""
    ent: Entity
    for ent in ctx.vmf.by_class['comp_case']:
        ent.remove()
        if check_control_enabled(ent):
            collapse_case(ctx, ent)
        else:
            # If any entities exist with the same name that aren't comp_case, we need to
            # keep the inputs.
            case_name = ent['targetname']
            if not any(ent['classname'].casefold() != 'comp_case' for ent in ctx.vmf.by_target[case_name]):
                ctx.add_io_remap_removal(case_name, 'InValue')
                ctx.add_io_remap_removal(case_name, 'Trigger')
                ctx.add_io_remap_removal(case_name, 'PickRandom')
