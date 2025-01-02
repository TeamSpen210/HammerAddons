"""Some keyvalues are case-sensitive, force the required casing."""
from srctools import VMF, logger

from hammeraddons.bsp_transform import Context, trans
from hammeraddons.bsp_transform.common import ent_description

LOGGER = logger.get_logger(__name__)


@trans('FGD - Fix key casing')
def force_case_sensitivity(ctx: Context) -> None:
    """Force case-sensitivity on some keyvalues that require it."""
    fix_casing(ctx.vmf, 'light_environment', 'SunSpreadAngle')
    fix_casing(ctx.vmf, 'light_directional', 'SunSpreadAngle')
    fix_casing(ctx.vmf, 'lua_run', 'Code')


def fix_casing(vmf: VMF, classname: str, *keys: str) -> None:
    """Fix the casing for one entity."""
    for ent in vmf.by_class[classname]:
        for key in keys:
            if value := ent.pop(key):
                LOGGER.warning(
                    'Correcting case of "{}" for {}',
                    key, ent_description(ent),
                )
                ent[key] = value
