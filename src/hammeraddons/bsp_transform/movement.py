"""Various transformations that move other entities around."""

from srctools import conv_int, conv_bool, Vec, conv_float
from srctools.bsp_transform import trans, Context
from srctools.logger import get_logger


LOGGER = get_logger(__name__)


@trans('comp_entity_mover')
def comp_entity_mover(ctx: Context):
    """Move an entity to another location."""
    for mover in ctx.vmf.by_class['comp_entity_mover']:
        mover.remove()

        ref_name = mover['reference']
        offset = Vec()
        if ref_name:
            for ent in ctx.vmf.search(ref_name):
                offset = Vec.from_str(mover['origin']) - Vec.from_str(ent['origin'])
                offset *= conv_float(mover['distance'])
                break
            else:
                LOGGER.warning(
                    'Can\'t find ref entity named "{}" '
                    'for comp_ent_mover at <{}>!',
                    ref_name,
                    mover['origin'],
                )
        else:
            # Use angles + movement.
            offset = Vec(x=conv_float(mover['distance']))
            offset = offset.rotate_by_str(mover['direction'])

        found_ent = None
        for found_ent in ctx.vmf.search(mover['target']):
            origin = Vec.from_str(found_ent['origin'])
            origin += offset
            found_ent['origin'] = str(origin)

        if found_ent is None:
            LOGGER.warning(
                'No entities found named "{}" for comp_ent_mover at ({})!',
                mover['target'],
                mover['origin'],
            )
