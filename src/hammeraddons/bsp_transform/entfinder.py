"""Implements comp_entity_finder."""
import itertools
from enum import Enum
from typing import Dict, Tuple

from srctools.bsp_transform import trans, Context
from srctools import Output, conv_bool, conv_float, Vec, Entity
from srctools.logger import get_logger

LOGGER = get_logger(__name__)


class FinderModes(Enum):
    CONST_TARG = 'const2target'
    CONST_KNOWN = 'const2known'
    KNOWN_TO_TARG = 'known2target'
    TARG_TO_KNOWN = 'target2known'
    OUTPUT_MERGE = 'replacetarget'

NEEDS = {
    # non-blank src, known ent
    FinderModes.CONST_TARG: (False, False),
    FinderModes.CONST_KNOWN: (False,  True),
    FinderModes.KNOWN_TO_TARG: (True, True),
    FinderModes.TARG_TO_KNOWN: (True, True),
    FinderModes.OUTPUT_MERGE: (False, True),
}


@trans('comp_entity_finder')
def entity_finder(ctx: Context):
    """Finds the closest entity of a given type."""
    target_cache = {}  # type: Dict[Tuple[str, float, float, float, float], Entity]

    for finder in ctx.vmf.by_class['comp_entity_finder']:
        finder.remove()
        targ_class = finder['targetcls']
        targ_radius = conv_float(finder['radius'])
        targ_ref = finder['targetref']

        targ_pos = Vec.from_str(finder['origin'])
        if targ_ref:
            for ent in ctx.vmf.search(targ_ref):
                targ_pos = Vec.from_str(ent['origin'])
                break
            else:
                LOGGER.warning(
                    'Can\'t find ref entity named "{}" '
                    'for entity finder at <{}>!',
                    targ_ref,
                    finder['origin'],
                )

        key = (targ_class, targ_radius) + targ_pos.as_tuple()
        try:
            found_ent = target_cache[key]
        except KeyError:
            found_ent = None
            cur_dist = float('inf')
            for targ_ent in ctx.vmf.by_class[targ_class]:
                dist_to = (Vec.from_str(targ_ent['origin']) - targ_pos).mag()
                if targ_radius == 0 or dist_to < targ_radius:
                    if cur_dist > dist_to:
                        found_ent = targ_ent
                        cur_dist = dist_to
            del targ_ent
            if found_ent is None:
                LOGGER.warning(
                    'Cannot find valid entity '
                    'for entity finder at <{}>!',
                    finder['origin'],
                )
                continue
            target_cache[key] = found_ent

        found_ent.outputs.extend(finder.outputs)
        finder.outputs.clear()

        # If the ent has no targetname, give it one.
        if not found_ent['targetname']:
            found_ent['targetname'] = found_ent['classname']
            found_ent.make_unique()

        for ind in itertools.count(1):
            kv_mode_str = finder['kv{}_mode'.format(ind)].casefold()
            if kv_mode_str == '':
                break

            try:
                kv_mode = FinderModes(kv_mode_str)
            except ValueError:
                LOGGER.warning(
                    'Unknown mode "{}" '
                    'for entity finder at <{}>!',
                    kv_mode_str,
                    finder['origin'],
                )
                continue

            kv_src = finder['kv{}_src'.format(ind)]
            kv_dest = finder['kv{}_dest'.format(ind)]

            # All modes need the destination keyvalue.
            if not kv_dest:
                LOGGER.warning(
                    'No destination keyvalue set '
                    'for entity finder at <{}>!',
                    finder['origin'],
                )
                continue

            needs_src, needs_known = NEEDS[kv_mode]

            known_ent = None
            if needs_known:
                for known_ent in ctx.vmf.search(finder['kv{}_known'.format(ind)]):
                    break
                if known_ent is None:
                    LOGGER.warning(
                        'Can\'t find known entity named '
                        '"{}" for entity finder at <{}>!',
                        known_ent,
                        finder['origin'],
                    )
                    continue

            if needs_src and not kv_src:
                LOGGER.warning(
                    'No source keyvalue set '
                    'for entity finder at <{}>!',
                    finder['origin'],
                )
                continue

            if kv_mode is FinderModes.CONST_TARG:
                # Set constant value on the found ent.
                found_ent[kv_dest] = kv_src
            elif kv_mode is FinderModes.CONST_KNOWN:
                # Set constant value on known entity.
                known_ent[kv_dest] = kv_src
            elif kv_mode is FinderModes.TARG_TO_KNOWN:
                known_ent[kv_dest] = found_ent[kv_src]
            elif kv_mode is FinderModes.KNOWN_TO_TARG:
                found_ent[kv_dest] = known_ent[kv_src]
            elif kv_mode is FinderModes.OUTPUT_MERGE:
                name = '!' + kv_dest.lstrip('!').casefold()
                for out in known_ent.outputs:
                    if out.target.casefold() == name:
                        out.target = found_ent['targetname']
            else:
                raise AssertionError('Unknown mode {}'.format(kv_mode))
