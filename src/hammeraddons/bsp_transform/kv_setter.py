"""Sets a keyvalue on an entity to a new value.

This is useful to compute spawnflags, or to adjust keyvalues when the target
entity's options can't be set to a fixup variable.
"""
from srctools import conv_int, conv_bool
from srctools.bsp_transform import trans, Context
from srctools.logger import get_logger

LOGGER = get_logger(__name__)


@trans('comp_kv_setter')
def kv_setter(ctx: Context) -> None:
    """Sets a keyvalue on an entity to a new value."""
    for setter in ctx.vmf.by_class['comp_kv_setter']:
        setter.remove()

        mode = setter['mode', 'kv']
        if mode.casefold() == 'flags':
            is_flags = True
        elif mode.casefold() == 'kv':
            is_flags = False
        else:
            LOGGER.warning(
                'Unknown mode ({}) for comp_kv_setter at ({})!',
                mode,
                setter['origin'],
            )
            continue

        kv_name = setter['kv_name']
        # Use fixup name if actually set.
        kv_value = setter['kv_value_local'] or setter['kv_value_global']

        if is_flags:
            try:
                # Convert using Python's literal rules,
                # to allow binary/hex literals.
                flag_mask = int(kv_name, base=0)
            except (TypeError, ValueError):
                LOGGER.warning(
                    'Invalid spawnflags mask for comp_kv_setter at ({})!\n'
                    'Provide an integer, 0x45 hex or 0b01101 binary value.',
                    mode,
                    setter['origin'],
                )
                continue
            flag_enabled = conv_bool(kv_value)
        else:
            flag_mask = 1
            flag_enabled = False

        found_ent = None

        for found_ent in ctx.vmf.search(setter['target']):
            if is_flags:
                spawnflags = conv_int(found_ent['spawnflags', '0'])
                if flag_enabled:
                    found_ent['spawnflags'] = spawnflags | flag_mask
                else:
                    found_ent['spawnflags'] = spawnflags & ~flag_mask
            else:
                found_ent[kv_name] = kv_value

        if found_ent is None:
            LOGGER.warning(
                'No entities found named "{}" for comp_kv_setter at ({})!',
                setter['target'],
                setter['origin'],
            )
