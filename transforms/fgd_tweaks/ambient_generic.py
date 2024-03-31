"""Allow ambient_generic's confusing spawnflags to be set via keyvalues instead."""
from typing import Final

from hammeraddons.bsp_transform import Context, trans, ent_description
from srctools import conv_int, logger


LOGGER = logger.get_logger(__name__)


FLAG_INFINITE: Final = 1
FLAG_START_SILENT: Final = 16
FLAG_NOT_LOOPED: Final = 32


@trans('FGD - ambient_generic keyvalues')
def ambient_generic_kvs(ctx: Context) -> None:
    """Allow ambient_generic's confusing spawnflags to be set via keyvalues instead.

    This also allows them to be easily set from $fixup values.
    Importantly this runs after inv_booleans, so it doesn't need to handle that.
    """
    for ent in ctx.vmf.by_class['ambient_generic']:
        flags = conv_int(ent['spawnflags'])

        # These are inverted.
        for name, pretty, flag in [
            ('haddons_enabled', 'Enabled', FLAG_START_SILENT),
            ('haddons_mode', 'Mode', FLAG_NOT_LOOPED),
        ]:
            value = conv_int(ent[name])
            if value == -1:
                pass  # Keep spawnflag
            elif value == 0:
                flags |= flag
            elif value == 1:
                flags &= ~flag
            else:
                LOGGER.warning(
                    '{} value "{}" is invalid for {}: must be -1, 0 or 1!',
                    pretty, ent[name], ent_description(ent),
                )

        # Non-inverted
        value = conv_int(ent['haddons_infrange'])
        if value == -1:
            pass  # Keep spawnflag
        elif value == 0:
            flags &= ~FLAG_INFINITE
        elif value == 1:
            flags |= FLAG_INFINITE
        else:
            LOGGER.warning(
                '{} value "{}" is invalid for {}: must be -1, 0 or 1!',
                'Infinite Range', ent['haddons_infrange'], ent_description(ent),
            )
        ent['spawnflags'] = flags
