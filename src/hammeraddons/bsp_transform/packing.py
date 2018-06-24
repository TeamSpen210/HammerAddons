"""Transformations for packing and precaching resources."""
from srctools.bsp_transform import trans, Context
from srctools.packlist import FileType
from srctools.logger import get_logger
from typing import Set
import os

LOGGER = get_logger(__name__, 'trans.packing')


@trans('comp_precache_model')
def comp_precache_model(ctx: Context):
    """Force precaching a specific model."""
    already_done = set()  # type: Set[str]
    for ent in ctx.vmf.by_class['comp_precache_model']:
        model = ent['model']
        if os.path.normcase(model) in already_done:
            ent.remove()
            continue
        already_done.add(os.path.normcase(model))

        model['classname'] = 'prop_dynamic_override'

        # Disable shadows and similar things on this to make it as cheap
        # as possible.
        model['rendermode'] = '10'
        model['disableshadowdepth'] = '1'
        model['disableshadows'] = '1'
        model['solid'] = '0'
        model['shadowdepthnocache'] = '2'
        model['spawnflags'] = '256'  # Start with collision off.

        # Move to a corner of the map, so it won't be in PVS generally.
        model['origin'] = '-15872 -15872 -15872'

# Keyvalue -> filetype.
PACK_TYPES = {
    'generic': FileType.GENERIC,
    'sound': FileType.GAME_SOUND,
    'model': FileType.MODEL,
    'material': FileType.MATERIAL,
    'particle': FileType.PARTICLE_FILE,
}


@trans('comp_pack')
def comp_pack(ctx: Context):
    """Force packing resources."""
    for ent in ctx.vmf.by_class['comp_pack']:
        ent.remove()
        for key, value in ent.keys.items():  # type: str, str
            # We allow numeric suffixes for multiple - generic45.
            try:
                res_type = PACK_TYPES[key.rstrip('0123456789').casefold()]
            except KeyError:
                LOGGER.warning(
                    'Unknown resource type: "{}" @ {}',
                    key,
                    ent['origin'],
                )
                res_type = FileType.GENERIC
            ctx.pack.pack_file(value, res_type)
