from typing import Iterable, List, Dict, Tuple
import itertools
import random
import re

from srctools import Vec, Entity, Output, conv_bool, conv_float, lerp
import srctools.logger

from hammeraddons.bsp_transform import trans, Context
from hammeraddons.bsp_transform.common import strip_cust_keys

LOGGER = srctools.logger.get_logger(__name__)
DIGIT_PATTERN = re.compile('[0-9]+')


def num_suffix(ent: Entity) -> int:
    """Find numbers at the end of the name, and then parse the number."""
    numbers = DIGIT_PATTERN.findall(ent['targetname'])
    if numbers:
        return int(numbers[-1])
    else:
        return 0

KEYVALUES = [
    'time_val', 'time_variance', 'time_mode',
    'order_mode', 'uniquify', 'target'
]


@trans('comp_sequential_call')
def sequential_call(ctx: Context) -> None:
    """Finds a sequence of entities (by distance or numeric suffix), then fires inputs delayed in order."""
    for seq_call in ctx.vmf.by_class['comp_sequential_call']:
        seq_call['classname'] = 'logic_relay'

        target_ents: List[Entity] = list(ctx.vmf.search(seq_call['target']))
        if not target_ents:
            LOGGER.warning(
                'Sequential call "{}" at {} could find no target entities named "{}"!',
                seq_call['targetname'], seq_call['origin'], seq_call['target'],
            )
            continue

        time_val = conv_float(seq_call['time_val'])
        time_variance = abs(conv_float(seq_call['time_variance']))
        time_mode = seq_call['time_mode'].casefold()
        order_mode = seq_call['order_mode'].casefold()
        make_unique = conv_bool(seq_call['uniquify'])
        origin = Vec.from_str(seq_call['origin'])

        if order_mode.startswith('dist'):
            dist_to_ent: Dict[Entity, float] = {
                ent: (Vec.from_str(ent['origin']) - origin).mag()
                for ent in target_ents
            }
            max_dist = max(dist_to_ent.values())
        else:
            dist_to_ent = {}
            max_dist = 1.0  # Ensure divide-by-zero check ignores this.

        if order_mode.startswith('dist'):
            target_ents.sort(key=dist_to_ent.__getitem__, reverse=order_mode.endswith('_inv'))
        elif order_mode.startswith('suffix'):
            target_ents.sort(key=num_suffix, reverse=order_mode.endswith('_inv'))
        else:
            raise ValueError(
                f'Unknown order mode "{order_mode}" for sequential call '
                f'"{seq_call["targetname"]}" at ({seq_call["origin"]}).'
            )

        ent_and_delay: Iterable[Tuple[Entity, float]]
        if max_dist < 1e-6 or time_val == 0.0:
            # No total delay, skip computation and any divide by zero.
            ent_and_delay = zip(target_ents, itertools.repeat(0.0))
        elif time_mode == 'total':
            time_start, time_end = (time_val, 0.0) if order_mode.endswith('_inv') else (0.0, time_val)
            # Special case, if total and dist selected, lerp by distance, not evenly spaced.
            if order_mode.startswith('dist'):
                ent_and_delay = (
                    (ent, lerp(dist_to_ent[ent], 0.0, max_dist, time_start, time_end))
                    for ent in target_ents
                )
            else:
                ent_and_delay = (
                    (ent, lerp(i, 0, len(target_ents), time_start, time_end))
                    for i, ent in enumerate(target_ents)
                )
        elif time_mode == 'interval':
            # [(ent, time_val * i) for i, ent in enumerate(target_ents)]
            ent_and_delay = zip(target_ents, map(time_val.__mul__, itertools.count()))
        else:
            raise ValueError(
                f'Unknown time mode "{time_mode}" for sequential call '
                f'"{seq_call["targetname"]}" at ({seq_call["origin"]}).'
            )

        outputs_rep: List[Output] = []
        outputs_final: List[Output] = []
        outputs_other: List[Output] = []
        for out in seq_call.outputs:
            out_name = out.output.casefold()
            if out_name == 'onseq':
                out.output = 'OnTrigger'
                outputs_rep.append(out)
            elif out_name == 'onseqend':
                out.output = 'OnTrigger'
                outputs_final.append(out)
            else:
                if out_name == 'onseqstart':
                    out.output = 'OnTrigger'
                outputs_other.append(out)
        seq_call.outputs[:] = outputs_other

        if not outputs_rep:
            LOGGER.warning(
                'Sequential call "{}" at ({}) has no OnSeq outputs, '
                "so there's nothing to repeat for each entity.",
                seq_call['targetname'], seq_call['origin'],
            )

        target = seq_call['target'].rstrip('*')
        max_delay = 0.0
        for ent, delay in ent_and_delay:
            if time_variance > 0.0:
                delay += random.uniform(-time_variance, time_variance)
            max_delay = max(max_delay, delay)
            if make_unique:
                ent.make_unique(seq_call['targetname'] + '_')
            for out in outputs_rep:
                out = out.copy()
                out.delay = round(out.delay + delay, 2)
                if out.target.casefold() == '!seq' or out.target == target:
                    out.target = ent['targetname']
                seq_call.outputs.append(out)
        for out in outputs_final:
            out.delay = round(out.delay + max_delay, 2)
            seq_call.outputs.append(out)

        strip_cust_keys(seq_call)
