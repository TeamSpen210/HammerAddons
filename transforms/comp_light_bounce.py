"""An entity that generates only baked bounced lighting."""

from hammeraddons.bsp_transform import trans, Context
from srctools.logger import get_logger
from srctools import VMF


@trans("comp_light_bounce", priority=999) # Run last to allow other transformations to take place first
def comp_light_bounce(ctx: Context):
    vmf: VMF = ctx.vmf

    for light in vmf.by_class["comp_light_bounce_spot"]:
        light["classname"] = "light_rt_spot"
        light["spawnflags"] = 1
        light["_lightmode"] = 2
    
    for light in vmf.by_class["comp_light_bounce"]:
        light["classname"] = "light_rt"
        light["spawnflags"] = 1
        light["_lightmode"] = 2


    