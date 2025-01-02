"""Allow boolean keyvalues to be inverted by prefixing them with !"""
from typing import Dict, Tuple

from srctools.fgd import EntityDef, ValueTypes
import srctools

from hammeraddons.bsp_transform import Context, trans


LOGGER = srctools.logger.get_logger(__name__)


@trans('FGD - Invertable Booleans', priority=-10)
def invertable_booleans(ctx: Context) -> None:
    """Look for !0/!1 in boolean keyvalues."""
    # classname, keyvalue -> is boolean.
    ent2is_bool: Dict[Tuple[str, str], bool] = {}

    for ent in ctx.vmf.entities:
        clsname = ent['classname']
        for key, value in ent.items():
            # Fetching the FGD type is more expensive than checking the value.
            # So do that first.
            if value not in ('!0', '!1'):
                continue
            try:
                is_bool = ent2is_bool[clsname.casefold(), key.casefold()]
            except KeyError:
                try:
                    kv_def = EntityDef.engine_def(clsname).kv[key]
                except KeyError:
                    # Unknown, assume it is invertible, and log this.
                    LOGGER.warning(
                        'FGD defintion for {}.{} not found, assuming it is boolean.',
                        clsname, key,
                    )
                    is_bool = True
                else:
                    is_bool = kv_def.type is ValueTypes.BOOL
                ent2is_bool[clsname.casefold(), key.casefold()] = is_bool
            if is_bool:
                ent[key] = '1' if value == '!0' else '0'
