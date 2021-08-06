"""Implements comp_entity_finder."""
import itertools
import math
from enum import Enum

from srctools.bsp_transform import trans, Context
from srctools import conv_bool, conv_float, Vec, Entity, Angle
from srctools.logger import get_logger

LOGGER = get_logger(__name__)


class FinderModes(Enum):
    """The kind of modification to apply to the found ent."""
    CONST_TARG = 'const2target'
    CONST_KNOWN = 'const2known'
    KNOWN_TO_TARG = 'known2target'
    TARG_TO_KNOWN = 'target2known'
    OUTPUT_MERGE = 'replacetarget'

NEEDS = {
    # For each mode, what data they need.
    # In order: a source keyvalue, and a known entity
    FinderModes.CONST_TARG: (False, False),
    FinderModes.CONST_KNOWN: (False,  True),
    FinderModes.KNOWN_TO_TARG: (True, True),
    FinderModes.TARG_TO_KNOWN: (True, True),
    FinderModes.OUTPUT_MERGE: (False, True),
}


@trans('comp_entity_finder')
def entity_finder(ctx: Context):
    """Finds the closest entity of a given type."""
    target_cache: dict[tuple, Entity] = {}

    for finder in ctx.vmf.by_class['comp_entity_finder']:
        finder.remove()
        targ_classes = frozenset(finder['targetcls'].split())
        targ_radius = conv_float(finder['radius'])
        targ_ref = finder['targetref']
        blacklist = finder['blacklist'].casefold()
        # Restrict to actually valid dot products.
        targ_fov = max(0.0, min(180.0, conv_float(finder['searchfov', '180.0'], 180.0)))

        # This will never find things, ignore that.
        if targ_fov == 0.0:
            LOGGER.warning(
                'Entity finder at <{}>! has FOV of 0, ignoring!',
                finder['origin'],
            )
            targ_fov = 180.0
        targ_dot = math.cos(math.radians(targ_fov))
        normal = Vec(x=1) @ Angle.from_str(finder['angles'])

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
        if len(targ_classes) == 0:
            LOGGER.warning(
                'Entity finder at <{}> has no '
                'classname specified.',
                finder['origin'],
            )
            continue

        key = (targ_classes, targ_radius, blacklist, targ_fov) + targ_pos.as_tuple()
        try:
            found_ent = target_cache[key]
        except KeyError:
            found_ent = None
            cur_dist = float('inf')
            targ_ent = None

            if blacklist.endswith('*'):
                blacklist = blacklist[:-1]

                def blacklist_func(name: str) -> bool:
                    """Check if the name matches the blacklist, with wildcards."""
                    return name.casefold().startswith(blacklist)
            elif blacklist:
                def blacklist_func(name: str) -> bool:
                    """Check if the name matches the blacklist exactly."""
                    return name.casefold() == blacklist
            else:
                def blacklist_func(name: str) -> bool:
                    """No blacklist."""
                    return False

            # If multiple, it's the union of the sets. If there's a single one
            # we don't need to copy by_class[].
            if len(targ_classes) == 1:
                [single_class] = targ_classes
                ent_set = ctx.vmf.by_class[single_class]
            else:
                ent_set = set.union(*[ctx.vmf.by_class[cls] for cls in targ_classes])

            for targ_ent in ent_set:
                if blacklist_func(targ_ent['targetname']):
                    continue

                offset = (Vec.from_str(targ_ent['origin']) - targ_pos)
                dist_to = offset.mag()

                # If at the same point, direction is meaningless.
                # Treat that as always passing the FOV check.
                if dist_to != 0.0 and Vec.dot(offset.norm(), normal) < targ_dot:
                    continue

                if targ_radius == 0 or dist_to < targ_radius:
                    if cur_dist > dist_to:
                        found_ent = targ_ent
                        cur_dist = dist_to
            del targ_ent
            if found_ent is None:
                # Convert the set of classes to a nice string.
                if len(targ_classes) == 0:
                    cls_desc = ''
                elif len(targ_classes) == 1:
                    [single_class] = targ_classes
                    cls_desc = single_class + ' '
                else:
                    [*first_classes, last_class] = sorted(targ_classes)
                    cls_desc = '{} or {}'.format(
                        ', '.join(first_classes),
                        last_class
                    )

                LOGGER.warning(
                    'Cannot find valid {}entity within {} units '
                    'for entity finder at <{}>! (fov={}, in direction {})',
                    cls_desc,
                    targ_radius,
                    finder['origin'],
                    targ_fov,
                    normal,
                )
                continue
            target_cache[key] = found_ent

        found_ent.outputs.extend(finder.outputs)
        finder.outputs.clear()

        # If the ent has no targetname, give it one.
        if not found_ent['targetname'] or conv_bool(finder['makeunique']):
            found_ent.make_unique('_found_entity')

        # If specified, teleport to the item's location.
        if conv_bool(finder['teleporttarget']):
            found_ent['origin'] = targ_pos

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
                    'for entity finder at <{}>, transformation #{}!',
                    finder['origin'],
                    ind,
                )
                continue

            needs_src, needs_known = NEEDS[kv_mode]

            known_ent = None
            if needs_known:
                known_ent_name = finder['kv{}_known'.format(ind)]
                if not known_ent_name:
                    LOGGER.warning(
                        'No known entity specified for entity finder at '
                        '<{}>, but one required for transformation #{}!',
                        finder['origin'],
                        ind,
                    )
                for known_ent in ctx.vmf.search(known_ent_name):
                    break
                if known_ent is None:
                    LOGGER.warning(
                        'Can\'t find known entity named '
                        '"{}" for entity finder at <{}>, transformation #{}!',
                        known_ent_name,
                        finder['origin'],
                        ind,
                    )
                    continue

            if needs_src and not kv_src:
                LOGGER.warning(
                    'No source keyvalue set '
                    'for entity finder at <{}>, transformation {}!',
                    finder['origin'],
                    ind,
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
