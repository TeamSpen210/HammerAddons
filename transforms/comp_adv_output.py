"""Adds a single output to an entity, with precise control over fixup behaviour.

"""
from typing import Any
from collections.abc import Collection, Mapping, Sequence
import itertools
import string
import struct

from srctools import EmptyMapping, Vec, conv_bool, conv_float, conv_int
from srctools.vmf import Output, Entity
from srctools.logger import get_logger

from hammeraddons.bsp_transform import trans, Context
from hammeraddons.bsp_transform.common import get_multimode_value, check_control_enabled, rng_get


class SimpleFormatter(string.Formatter):
    """Use 1-based indexes for the args, instead of 0-based."""
    def get_value(self, key: int | str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
        """Adjust the key."""
        if isinstance(key, int) and key > 0:
            return args[key - 1]
        raise KeyError(key)


LOGGER = get_logger(__name__)
FORMATTER = SimpleFormatter()


@trans('comp_adv_output')
def advanced_output(ctx: Context) -> None:
    """Adds a single output to an entity, with precise control over fixup behaviour."""
    adv_out: Entity
    for adv_out in ctx.vmf.by_class['comp_adv_output']:
        adv_out.remove()
        if not check_control_enabled(adv_out):
            continue

        output_name = adv_out['out_name']
        input_name = adv_out['inp_name']
        target_name = adv_out['target_local'] or adv_out['target_global']
        times = conv_int(adv_out['times'], -1)
        expand_target = conv_bool(adv_out['target_expand'])
        if not target_name:
            LOGGER.warning(
                'No target entity for conv_adv_output at ({})!',
                adv_out['origin'],
            )
            continue

        if target_instname := adv_out['target_instname']:
            target_name = f'{target_name}-{target_instname}'

        delay = conv_float(adv_out['delay'])

        delay_max = adv_out['delay_max']
        if delay_max:
            rng = rng_get('comp_adv_output', adv_out)
            delay = rng.uniform(delay, conv_float(delay_max, delay))

        delay += conv_float(adv_out['delay2'])
        if delay < 0.0:
            LOGGER.warning(
                'conv_adv_output at ({}) has a negative delay!',
                adv_out['origin'],
            )
            delay = 0.0

        param_args: list[str] = []
        for ind in itertools.count(1):
            val = get_multimode_value(adv_out, prefix='params_', suffix=str(ind), desc=f'Param {ind}')
            if not val:
                break
            param_args.append(val)

        parameter = adv_out['params_fmt']
        if parameter:
            try:
                parameter = FORMATTER.vformat(parameter, param_args, EmptyMapping)
            except Exception:
                LOGGER.exception(
                    'Failed to format comp_adv_output at ({}) using format "{}" and parameters {!r}',
                    adv_out['origin'], parameter, param_args
                )
                continue

        found_ent = None
        targets: Collection[str] = (target_name, )
        for found_ent in ctx.vmf.search(adv_out['out_ent']):
            if expand_target:
                targets = {
                    targ['targetname']
                    for targ in ctx.vmf.search(target_name)
                }
                if not any(targets):
                    LOGGER.warning(
                        'No matching named entities for search "{}" for conv_adv_output at ({})!',
                        target_name
                    )
                    # Fall back to letting the game do the search.
                    targets = (target_name, )
                else:
                    LOGGER.debug(
                        'Expanding {} -> {} = {}',
                        found_ent['targetname'], target_name,
                        sorted(targets),
                    )
            for targ in targets:
                found_ent.add_out(Output(
                    output_name,
                    targ,
                    input_name,
                    parameter,
                    delay,
                    times=times,
                ))

        if found_ent is None:
            LOGGER.warning(
                'No entities found named "{}" for comp_adv_output at ({})!',
                adv_out['target'], adv_out['origin'],
            )
